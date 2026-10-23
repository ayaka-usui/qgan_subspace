# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs of a multiple experiment trained together (CFG.parallel_runs = True), and the batched training step they use.

BatchedQWGAN      R independent trainings of one circuit (Choi mode, local critic, the spectral evaluation) stepped
                  together: one broadcast circuit call for all the generators (PennyLane parameter broadcasting), one
                  call of the exact eigenbasis formula for all the critic gradients (qgan.dis_spectral), and one SGD
                  step on the stacked parameters (elementwise, so each run's update is its own).
ParallelTraining  the runs of one phase of tools.training_init (its specs), as many at a time as fit in memory
                  (max_runs_together): a run that stops (at the end of an epoch, as alone) leaves the batch, and the
                  next run of the phase takes its place. Everything else is the per-run code of qgan.training.Training
                  (_begin, _after_iteration, _end_epoch, _finish): the same logs, plots, histories, saved models and
                  files as when each run is trained alone, and the same results up to rounding errors (the batched sums
                  can round differently).

Random numbers: a run draws its torch random numbers (initial parameters and critic, inserted ancilla angles, warm
starts) when it is built, and the runs are built in their order, so they are those of the runs trained one after the
other. The exception is compute_entanglement = True: the random product inputs of the negativities come from numpy's
global generator, drawn here run after run at every evaluation instead of one run's whole training after the other's,
so each run gets other inputs (from the same distribution), and the numpy draws of the runs' loading
(type_of_warm_start = "some": which parameters are re-drawn) move with them.

Needs use_choi = True and dis_eval = "spectral" (local critic): config._validate. The per-epoch plots are drawn in
background processes, so that they do not hold up the training of the other runs.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from typing import Optional

import torch

from config import CFG
from qgan.ancilla import gen_branches_batched
from qgan.cost_functions import compute_fidelity
from qgan.discriminator import capped_rows
from qgan.dis_spectral import DisSpectrum, critic_gradients, spectral_wasserstein
from qgan.precision import cdtype
from qgan.training import Training
from tools.data_managers import print_and_log, print_and_log_with_headers
from tools.plot_hub import plt_fidelity_vs_iter

# Share of the available memory that the runs trained together may use (max_runs_together)
MEMORY_FRACTION = 0.5
# Runs trained together when the available memory cannot be read
FALLBACK_RUNS = 16


# =====================================================================================================================
# The batched training step
# =====================================================================================================================
def _leaf(x, dtype) -> torch.Tensor:
    return torch.as_tensor(x).detach().to(dtype).clone().requires_grad_(True)


class BatchedQWGAN:
    """R independent QWGAN trainings of one circuit, stepped together (Training._train_step_choi for each run).

    Args:
        circuit, choi_input, target: the generator's QNode (Generator.circuit), its Choi input state and the target
            state seen by the critic (Training.final_target_state); CFG must hold the circuit's settings when the
            methods run (the ansatz and the ancilla post-processing read CFG).
        theta (R, P), alpha (R, n, 4), beta (R, n, 4): the runs' parameters.
        sgd_gen, sgd_dis: the torch.optim.SGD settings of the generator's and the critic's optimiser (lr, momentum,
            and optionally dampening, weight_decay, nesterov).
        buffers: (generator, alpha, beta) SGD momentum buffers stacked over the runs, or None (zeros: as a new
            optimiser, the first update is the same).
        freeze: the critic's last row (its ancilla row) keeps its value (dis_ancilla_row_trainable = False).
        mask: (P,) bool, False for the generator parameters that are not trained (ancilla_training = False).
        cap: CFG.dis_module_cap of the critics (None: no cap), applied after every critic step.
        states: (R, D) generator outputs of theta, if known (else computed with one broadcast call).
    """

    def __init__(self, circuit, choi_input: torch.Tensor, target: torch.Tensor, theta, alpha, beta, *,
                 rdt: torch.dtype, sgd_gen: dict, sgd_dis: dict, buffers: Optional[tuple] = None, freeze: bool = False,
                 mask: Optional[torch.Tensor] = None, cap: Optional[float] = None,
                 states: Optional[torch.Tensor] = None):
        self.circuit, self.choi_input, self.target, self.rdt = circuit, choi_input, target, rdt
        self.sgd_gen, self.sgd_dis = dict(sgd_gen), dict(sgd_dis)
        self.freeze, self.mask, self.cap = freeze, mask, cap
        self.theta, self.alpha, self.beta = _leaf(theta, rdt), _leaf(alpha, rdt), _leaf(beta, rdt)
        self._make_optimizers(buffers)
        self.states = self.circuit_states(self.theta.detach()) if states is None else states.detach()

    # -- the stacked tensors and their optimisers ------------------------------------------------------------------
    def _make_optimizers(self, buffers: Optional[tuple]) -> None:
        self.opt_g = torch.optim.SGD([self.theta], **self.sgd_gen)
        self.opt_d = torch.optim.SGD([self.alpha, self.beta], **self.sgd_dis)
        if buffers is None:
            return
        for opt, p, buf in ((self.opt_g, self.theta, buffers[0]), (self.opt_d, self.alpha, buffers[1]),
                            (self.opt_d, self.beta, buffers[2])):
            if opt.param_groups[0]["momentum"] != 0:
                opt.state[p]["momentum_buffer"] = buf.detach().clone()

    def buffers(self) -> tuple:
        """The SGD momentum buffers (generator, alpha, beta), zeros before the first step."""
        out = []
        for opt, p in ((self.opt_g, self.theta), (self.opt_d, self.alpha), (self.opt_d, self.beta)):
            buf = opt.state.get(p, {}).get("momentum_buffer")
            out.append(torch.zeros_like(p.detach()) if buf is None else buf.detach())
        return tuple(out)

    @property
    def size(self) -> int:
        return self.theta.shape[0]

    def keep(self, rows: list[int]) -> None:
        """Only the runs `rows` stay (in that order)."""
        idx = torch.tensor(rows, dtype=torch.long)
        buffers = tuple(b[idx] for b in self.buffers())
        self.theta, self.alpha, self.beta = (_leaf(p.detach()[idx], self.rdt) for p in (self.theta, self.alpha,
                                                                                          self.beta))
        self._make_optimizers(buffers)
        self.states = self.states[idx]

    def append(self, theta, alpha, beta, buffers: tuple, states: torch.Tensor) -> None:
        """More runs (their parameters, momentum buffers and generator outputs) after the current ones."""
        old = self.buffers()
        self.theta, self.alpha, self.beta = (_leaf(torch.cat([p.detach(), torch.as_tensor(q).to(self.rdt)]), self.rdt)
                                             for p, q in ((self.theta, theta), (self.alpha, alpha), (self.beta, beta)))
        self._make_optimizers(tuple(torch.cat([o, b.to(o.dtype)]) for o, b in zip(old, buffers)))
        self.states = torch.cat([self.states, states.detach()])

    # -- one iteration ---------------------------------------------------------------------------------------------
    def _theta(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        th = self.theta if theta is None else theta
        return th if self.mask is None else torch.where(self.mask, th, th.detach())

    def circuit_states(self, theta: torch.Tensor) -> torch.Tensor:
        """(R, D) generator outputs of theta (R, P), one broadcast circuit call (with the autograd graph if theta
        requires it)."""
        th = self._theta(theta)
        if not th.requires_grad:
            with torch.no_grad():
                return self.circuit(th.T, self.choi_input)
        return self.circuit(th.T, self.choi_input)

    def critic_states(self) -> torch.Tensor:
        """What the critics see: (R, d) states, or (R, 2, d) branches (ancilla_mode "trace")."""
        return gen_branches_batched(self.states)

    @torch.no_grad()
    def _cap(self) -> None:
        """The critic norm cap, run by run as Discriminator.cap_norm (a frozen ancilla row keeps its value)."""
        if self.cap is None:
            return
        for p in (self.alpha, self.beta):
            n_scaled = p.shape[1] - 1 if self.freeze else None
            norm = (p[..., 0].abs() + p[..., 1:].norm(dim=-1)).prod(dim=1)
            for k in torch.nonzero(norm > self.cap * (1 - 1e-9)).flatten().tolist():  # capped_rows decides exactly
                p[k] = capped_rows(p[k], self.cap, n_scaled)

    def critic_step(self) -> None:
        """One critic step at the current (detached) generator outputs (the optimiser minimises -W; a frozen row
        keeps its value)."""
        self.opt_d.zero_grad()
        G = self.critic_states().detach()
        da, db = critic_gradients(G.cpu().numpy(), self.target.cpu().numpy(), self.alpha.detach().cpu().numpy(),
                                  self.beta.detach().cpu().numpy())
        self.alpha.grad = torch.tensor(-da, dtype=self.rdt)  # the critic maximises W: the gradient of -W
        self.beta.grad = torch.tensor(-db, dtype=self.rdt)
        kept = None
        if self.freeze:
            self.alpha.grad[:, -1] = 0.0
            self.beta.grad[:, -1] = 0.0
            kept = (self.alpha.detach()[:, -1].clone(), self.beta.detach()[:, -1].clone())
        self.opt_d.step()
        if kept is not None:
            with torch.no_grad():
                self.alpha[:, -1], self.beta[:, -1] = kept
        self._cap()

    def spectrum(self) -> DisSpectrum:
        return DisSpectrum(self.alpha.detach().cpu().numpy(), self.beta.detach().cpu().numpy(), self.rdt, cdtype())

    def costs(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(R,) W of each run's generator (theta, or its current parameters) with its critic held fixed."""
        total = self.states if theta is None else self.circuit_states(theta)
        return spectral_wasserstein(gen_branches_batched(total), self.target, self.spectrum())

    def generator_step(self) -> None:
        """One generator step with the current (detached) critics (theta.grad keeps the gradient used)."""
        self.opt_g.zero_grad()
        self.costs(self.theta).sum().backward()
        self.opt_g.step()
        self.states = self.circuit_states(self.theta.detach())


# =====================================================================================================================
# How many runs fit
# =====================================================================================================================
def available_memory() -> Optional[int]:
    """Bytes of memory available now (psutil, /proc/meminfo or sysconf), or None if none of them tells."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def bytes_per_run(engine: BatchedQWGAN) -> int:
    """Memory of one run's iteration: the tensors its generator step keeps for the backward pass (measured on the
    engine's first run), times 3 for the gradients and temporaries of the backward pass."""
    sizes = {}

    def pack(t):
        try:
            st = t.untyped_storage()
            sizes[(st.data_ptr(), st.nbytes())] = st.nbytes()
        except Exception:
            sizes[id(t)] = t.numel() * t.element_size()
        return t

    th = engine.theta[:1].detach().clone().requires_grad_(True)
    spectrum = DisSpectrum(engine.alpha[:1].detach().cpu().numpy(), engine.beta[:1].detach().cpu().numpy(),
                           engine.rdt, cdtype())
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        total = engine.circuit_states(th)
        spectral_wasserstein(gen_branches_batched(total), engine.target, spectrum).sum()
    return 3 * (sum(sizes.values()) + 4 * total.numel() * total.element_size())


def max_runs_together(engine: BatchedQWGAN, n_runs: int) -> tuple[int, str]:
    """The number of runs to train together (of n_runs): all of them if they fit in MEMORY_FRACTION of the available
    memory, else as many as fit (at least 1). Training more runs together never costs time (the time per iteration
    grows with the number of runs, the time per run falls), so the only limit is memory. Returns (number, note)."""
    per_run = bytes_per_run(engine)
    avail = available_memory()
    if avail is None:
        n = min(n_runs, FALLBACK_RUNS)
        return n, f"{n} of {n_runs} (available memory unknown; {per_run / 2**20:.0f} MB per run)"
    n = int(max(1, min(n_runs, MEMORY_FRACTION * avail // max(per_run, 1))))
    return n, f"{n} of {n_runs} ({per_run / 2**20:.1f} MB per run, {avail / 2**30:.1f} GB available)"


# =====================================================================================================================
# The runs of a phase
# =====================================================================================================================
def apply_run(spec: dict) -> None:
    """Point CFG to one run: its folder (base_data_path, and the paths of its files) and its load_timestamp."""
    CFG.load_timestamp = spec.get("load_timestamp")
    CFG.base_data_path = spec["base_data_path"]
    CFG.set_results_paths()


def _plot_workers() -> int:
    return max(1, min(2, (os.cpu_count() or 2) - 1))


_PLOT_POOL = None  # one pool of plotting processes for all the runs of a multiple experiment (started once)


def _submit_plot(*args):
    """plt_fidelity_vs_iter(*args) in the background pool (started on first use; restarted if a worker died)."""
    global _PLOT_POOL
    for attempt in range(2):
        if _PLOT_POOL is None:
            _PLOT_POOL = ProcessPoolExecutor(max_workers=_plot_workers(), mp_context=multiprocessing.get_context("spawn"))
        try:
            return _PLOT_POOL.submit(plt_fidelity_vs_iter, *args)
        except BrokenProcessPool:
            _PLOT_POOL = None
            if attempt:
                raise


def _sgd_settings(optimizer) -> dict:
    keys = ("lr", "momentum", "dampening", "weight_decay", "nesterov")
    return {k: optimizer.param_groups[0][k] for k in keys}


def _momentum(optimizer, param) -> torch.Tensor:
    buf = optimizer.state.get(param, {}).get("momentum_buffer")
    return torch.zeros_like(param.detach()) if buf is None else buf.detach().clone()


class ParallelTraining:
    """The runs of `specs` trained together, as many at a time as fit in memory: each spec is a dict with
    base_data_path, load_timestamp (or None) and optionally header / done (messages logged in that run's log before
    it starts / after it finishes)."""

    def __init__(self, specs: list[dict]):
        if not (CFG.use_choi and CFG.dis_eval == "spectral" and CFG.dis_type == "local"):
            raise ValueError("Runs are trained together only with use_choi=True and dis_eval='spectral' (local critic).")
        self.specs = list(specs)
        self.n_started = 0  # specs started so far (in their order)
        self.active: list[tuple[int, Training]] = []  # (spec index, run) of the runs in the batch, in batch order
        self.engine: Optional[BatchedQWGAN] = None
        self.capacity: Optional[int] = None
        self._plots = []

    # -- runs joining and leaving ----------------------------------------------------------------------------------
    def _start(self, i: int) -> Training:
        spec = self.specs[i]
        apply_run(spec)
        if spec.get("header"):
            print_and_log_with_headers(spec["header"], CFG.log_path)
        training = Training()
        training._begin()  # configuration log, loading of its models, histories
        return training

    def _rows(self, runs: list[Training]) -> dict:
        """The engine inputs of these runs: parameters, momentum, generator outputs; their settings checked."""
        for tr in runs:
            settings = (_sgd_settings(tr.gen.optimizer), _sgd_settings(tr.dis.optimizer))
            if self.engine is not None and settings != (self.engine.sgd_gen, self.engine.sgd_dis):
                raise ValueError("The runs trained together have different optimiser settings (lr, momentum, ...): "
                                 "a run loaded its optimiser from a file of other settings (reset_optimizer_on_load "
                                 "= True gives every run the current ones).")
        return dict(theta=torch.stack([tr.gen.params.detach() for tr in runs]),
                    alpha=torch.stack([tr.dis.alpha.detach() for tr in runs]),
                    beta=torch.stack([tr.dis.beta.detach() for tr in runs]),
                    buffers=(torch.stack([_momentum(tr.gen.optimizer, tr.gen.params) for tr in runs]),
                             torch.stack([_momentum(tr.dis.optimizer, tr.dis.alpha) for tr in runs]),
                             torch.stack([_momentum(tr.dis.optimizer, tr.dis.beta) for tr in runs])),
                    states=torch.stack([tr.gen.total_gen_state.detach() for tr in runs]))

    def _fill(self) -> None:
        """Start runs (in their order) until the batch is full or every run has started."""
        if self.engine is None:  # the first run sets the circuit, the settings and the batch size
            first = self._start(self.n_started)
            self.n_started += 1
            self.active.append((self.n_started - 1, first))
            gen, dis = first.gen, first.dis
            mask = None
            if gen.ancilla and not CFG.ancilla_training:  # (as Generator._effective_params)
                mask = torch.ones(gen.n_params, dtype=torch.bool)
                mask[gen._get_ancilla_param_indices()] = False
            rows = self._rows([first])
            self.engine = BatchedQWGAN(
                gen.circuit, gen.choi_input, first.final_target_state, rows["theta"], rows["alpha"], rows["beta"],
                rdt=gen.params.dtype, sgd_gen=_sgd_settings(gen.optimizer), sgd_dis=_sgd_settings(dis.optimizer),
                buffers=rows["buffers"], freeze=not CFG.dis_ancilla_row_trainable and dis.ancilla
                and dis.ancilla_mode == "pass", mask=mask, cap=CFG.dis_module_cap, states=rows["states"])
            self.capacity, note = max_runs_together(self.engine, len(self.specs))
            print(f"Runs trained together: {note}", flush=True)
        new = []
        while len(self.active) + len(new) < self.capacity and self.n_started < len(self.specs):
            new.append((self.n_started, self._start(self.n_started)))
            self.n_started += 1
        if new:
            rows = self._rows([tr for _, tr in new])
            self.engine.append(rows["theta"], rows["alpha"], rows["beta"], rows["buffers"], rows["states"])
            self.active += new

    def _sync(self, ks: list[int], final: bool) -> None:
        """Copy the batched parameters and outputs back to the runs at batch positions ks (final: also the critics and
        the momentum of both optimisers, for saving)."""
        e = self.engine
        buffers = e.buffers() if final else None
        with torch.no_grad():
            for k in ks:
                tr = self.active[k][1]
                tr.gen.params.copy_(e.theta[k])
                tr.gen.total_gen_state = e.states[k].clone()
                if not final:
                    continue
                tr.dis.alpha.copy_(e.alpha[k])
                tr.dis.beta.copy_(e.beta[k])
                for opt_run, p_run, buf in ((tr.gen.optimizer, tr.gen.params, buffers[0]),
                                            (tr.dis.optimizer, tr.dis.alpha, buffers[1]),
                                            (tr.dis.optimizer, tr.dis.beta, buffers[2])):
                    if opt_run.param_groups[0]["momentum"] != 0:
                        opt_run.state[p_run]["momentum_buffer"] = buf[k].clone()

    # -- one iteration of every run (as Training._train_step_choi) -----------------------------------------------------
    def _step(self) -> None:
        for _ in range(CFG.steps_dis):
            self.engine.critic_step()
        for _ in range(CFG.steps_gen):
            self.engine.generator_step()

    @torch.no_grad()
    def _evaluate(self) -> list[tuple]:
        """(fidelity, cost, total state) of every run, as Training._evaluate_choi."""
        G = self.engine.critic_states()
        W = spectral_wasserstein(G, self.engine.target, self.engine.spectrum())
        return [(compute_fidelity(self.engine.target, G[k]), W[k].item(), self.engine.states[k])
                for k in range(G.shape[0])]

    # -- per-epoch plots in background processes ----------------------------------------------------------------------
    def _plot(self, fidelities, losses, config, *rest) -> None:
        """plt_fidelity_vs_iter in a worker process, with the current run's paths (CFG points to it)."""
        snapshot = SimpleNamespace(compute_entanglement=config.compute_entanglement, figure_path=config.figure_path,
                                   system_size=config.system_size, gen_layers=config.gen_layers)
        self._plots.append(_submit_plot(list(fidelities), list(losses), snapshot, *rest))

    def _wait_for_plots(self) -> None:
        plots, self._plots = self._plots, []
        for future in plots:
            future.result()

    # -- the training loop (as Training.run, for all runs) --------------------------------------------------------------
    def run(self) -> None:
        try:
            while True:
                self._fill()
                if not self.active:
                    break
                for i, tr in self.active:
                    apply_run(self.specs[i])
                    tr._start_epoch()
                for epoch_iter in range(CFG.iterations_epoch):
                    self._step()
                    if epoch_iter % CFG.save_fid_and_loss_every_x_iter == 0:  # (the log lines are a subset)
                        evaluations = self._evaluate()
                        if CFG.compute_entanglement:  # its diagnostics read the generators' parameters and outputs
                            self._sync(list(range(len(self.active))), final=False)
                        for k, (i, tr) in enumerate(self.active):
                            apply_run(self.specs[i])
                            tr._after_iteration(epoch_iter, evaluations[k])
                stopped = []
                for k, (i, tr) in enumerate(self.active):
                    apply_run(self.specs[i])
                    if tr._end_epoch(plotter=self._plot):
                        stopped.append(k)
                if stopped:
                    self._sync(stopped, final=True)
                    for k in stopped:
                        i, tr = self.active[k]
                        apply_run(self.specs[i])
                        tr._finish()
                        if self.specs[i].get("done"):
                            print_and_log(self.specs[i]["done"], CFG.log_path)
                    keep = [k for k in range(len(self.active)) if k not in stopped]
                    self.active = [self.active[k] for k in keep]
                    self.engine.keep(keep)
        finally:
            self._wait_for_plots()
        apply_run(self.specs[-1])  # CFG left pointing to the last run, as after running them one after the other
