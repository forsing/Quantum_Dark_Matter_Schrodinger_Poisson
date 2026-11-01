#!/usr/bin/env python3

# Quantum_Dark_Matter_Schrodinger_Poisson


"""
Quantum_Dark_Matter
Schrodinger-Poisson

Osnovna dinamika:
    i d(psi_r)/dt =
        lambda_r * L psi_r / 2
        + (V + beta * observed_drive) psi_r / lambda_r

    -L V = gravity * (39 * rho - mean(39 * rho))

    rho = sum_r mass_r * abs(psi_r)**2 / sum_r mass_r
    mass_r = 1 / lambda_r

L je Laplasijan kauzalnog grafa 39 simbola.

Graf:
    Formira se iz kovarijansi zajednickog pojavljivanja.
    Koristi samo do tada poznata kola.
    Veze nisu rangiranje frekvencija za NEXT.
    Brojevi nemaju unapred nametnuto prostorno rastojanje.

Qiskit:
    Svako polje koristi sest kubita.
    Razvoj izvrsava Statevector.evolve sa kvantnim operatorima.
    Potencijal se klasicno azurira iz trenutne kvantne gustine.
    Nelinearnost je hibridna; nije jedan fiksan unitar.

Zajednicka amplituda:
    A(S) = sum_r c_r * product_{j in S} psi_r[j]

    P(S) = abs(A(S))**2 / Z
    S je skup sedam razlicitih brojeva.

Koeficijenti c_r nastaju izvrsavanjem dvokubitnog Qiskit kola.
Interferencija je izmedju amplituda celih kombinacija.

Sedam sestokubitnih izlaznih registara ima kompaktnu
reprezentaciju kroz tri koherentne grane.
Ne alocira se puni vektor stanja od 42 kubita.

Amplituda izabrane kombinacije dodatno se proverava
Qiskit kolom sa dva kontrolna i sedam pomocnih kubita.

Klasicni deo:
    Konstrukcija grafa.
    Poissonov potencijal.
    Optimizacija cetiri ugla kvantnog koherentnog ocitavanja.
    Tacna kontrakcija normalizatora i marginala.
    Iscrpna pretraga NEXT.

Bez klasifikatora, slucajnih kombinacija, uzorkovanja,
zaokruzivanja, popravljanja duplikata i kozmetickih filtera.

Pocetno simetricno polje je pocetni uslov.
Izlazna raspodela razvija se kroz podatke i ucenje.

Originalna kosmoloska prostorna mreza ovde je zamenjena grafom.
Posmatrana kola uvode spoljasnji potencijal.
Koherentno ocitavanje kombinacije je deo Loto adaptacije.
"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import csv
import hashlib
import itertools
import math
import platform
import random
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np

random.seed(39)
np.random.seed(39)

import qiskit

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import minimize


CSV_PATHS = (
    Path("/data/loto7_4712_k87.csv"),
    Path("/data/loto7_4712_k87_loto_2978.csv"),
    Path("/data/loto7_4712_k87_loto_plus_1734.csv"),
)


NUMBERS = 39
DRAW_SIZE = 7
FIELDS = 3
QUBITS = 6
DIM = 64

SUBSTEPS = 4
SEARCH_CHUNK = 65536

MAX_ITERATIONS = 100
MAX_EVALUATIONS = 180
GRADIENT_TOLERANCE = 1.0e-6

# Odbija numericki gotovo potpuno ponistena kvantna stanja.
# Ne dodaje masu raspodeli i ne menja njene verovatnoce.
MIN_RELATIVE_NORM = 1.0e-8

BOUNDS = (
    (0.0, np.pi),
    (0.0, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
)

STARTS = (
    np.array([1.0, 1.3, 0.0, 0.0]),
    np.array([1.6, 1.0, 0.7, -0.4]),
    np.array([2.3, 1.8, -1.0, 0.5]),
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


class InvalidState(ArithmeticError):
    pass


def number_text(values):
    return " ".join(str(int(value)) for value in values)


@dataclass(frozen=True)
class Physics:
    name: str
    wavelength: float
    gravity: float
    drive: float

    @property
    def lambdas(self):
        return self.wavelength * np.array([0.6, 1.0, 1.7])


PHYSICS = (
    Physics("SP_A", 0.5, 0.005, 0.10),
    Physics("SP_B", 0.9, 0.010, 0.18),
    Physics("SP_C", 1.3, 0.020, 0.26),
)


class Spectrum:
    def __init__(self, values, vectors):
        self.values = np.asarray(values, dtype=np.float64)
        self.vectors = np.asarray(vectors, dtype=np.float64)

        require(
            abs(float(self.values[0])) < 1.0e-10
            and float(self.values[1]) > 1.0e-8,
            "Graf nema ocekivanu povezanu Poissonovu strukturu.",
        )

        self.inverse = np.zeros(NUMBERS, dtype=np.float64)
        self.inverse[1:] = 1.0 / self.values[1:]
        self.cache = {}

    def laplacian(self, vector):
        return self.vectors @ (
            self.values * (self.vectors.T @ vector)
        )

    def potential(self, density, gravity):
        contrast = NUMBERS * (
            density - float(np.mean(density))
        )

        result = self.vectors @ (
            -gravity
            * self.inverse
            * (self.vectors.T @ contrast)
        )

        return result - float(np.mean(result))

    def kinetics(self, lambdas, dt):
        key = (
            tuple(float(value) for value in lambdas),
            float(dt),
        )

        if key not in self.cache:
            matrices = []

            for wavelength in lambdas:
                phases = np.exp(
                    -0.5j * dt * wavelength * self.values
                )

                physical = (
                    self.vectors * phases[None, :]
                ) @ self.vectors.T

                matrix = np.eye(DIM, dtype=np.complex128)
                matrix[:NUMBERS, :NUMBERS] = physical
                matrices.append(matrix)

            self.cache[key] = matrices

        return self.cache[key]


class GraphHistory:
    def __init__(self, draws, announce=False):
        self.count = len(draws)

        self.values = np.empty(
            (self.count, NUMBERS),
            dtype=np.float64,
        )

        self.vectors = np.empty(
            (self.count, NUMBERS, NUMBERS),
            dtype=np.float64,
        )

        sums = np.zeros(NUMBERS, dtype=np.float64)
        products = np.zeros(
            (NUMBERS, NUMBERS),
            dtype=np.float64,
        )

        for origin, draw in enumerate(draws):
            observed = np.zeros(NUMBERS, dtype=np.float64)
            observed[draw - 1] = 1.0

            sums += observed
            products += np.outer(observed, observed)

            mean = sums / (origin + 1)

            covariance = (
                products / (origin + 1)
                - np.outer(mean, mean)
            )

            weights = (
                1.0 + NUMBERS * np.abs(covariance)
            )

            np.fill_diagonal(weights, 0.0)

            laplacian = (
                np.diag(np.sum(weights, axis=1))
                - weights
            ) / NUMBERS

            laplacian = 0.5 * (
                laplacian + laplacian.T
            )

            values, vectors = np.linalg.eigh(laplacian)

            require(
                abs(float(values[0])) < 1.0e-10
                and float(values[1]) > 1.0e-8,
                "Kauzalni graf nije povezan.",
            )

            values[0] = 0.0

            self.values[origin] = values
            self.vectors[origin] = vectors

            if announce and (origin + 1) % 1000 == 0:
                print(
                    f"Kauzalni graf:"
                    f" {origin + 1}/{self.count}"
                )

    def at(self, origin):
        return Spectrum(
            self.values[origin],
            self.vectors[origin],
        )


def quantum_apply(state, matrix, as_circuit=False):
    if as_circuit:
        circuit = QuantumCircuit(QUBITS)
        circuit.unitary(
            matrix,
            list(range(QUBITS)),
        )

        return Statevector(state).evolve(circuit).data

    return Statevector(state).evolve(
        Operator(matrix)
    ).data


class SPEvolver:
    def __init__(self, physics):
        self.physics = physics
        self.lambdas = physics.lambdas
        self.masses = 1.0 / self.lambdas

    def initial_state(self):
        initial = np.zeros(DIM, dtype=np.complex128)
        initial[:NUMBERS] = 1.0 / np.sqrt(NUMBERS)

        circuit = QuantumCircuit(QUBITS)
        circuit.initialize(initial)

        state = Statevector.from_instruction(circuit).data

        return np.repeat(
            state[None, :],
            FIELDS,
            axis=0,
        )

    def density(self, states):
        return np.sum(
            self.masses[:, None]
            * np.abs(states[:, :NUMBERS]) ** 2,
            axis=0,
        ) / float(np.sum(self.masses))

    def advance(
        self,
        states,
        drive,
        spectrum,
        steps=SUBSTEPS,
        duration=1.0,
        gravity=None,
        as_circuit=False,
    ):
        require(
            steps >= 1 and np.isfinite(duration),
            "Neispravan vremenski korak.",
        )

        if gravity is None:
            gravity = self.physics.gravity

        dt = float(duration) / steps
        kinetics = spectrum.kinetics(self.lambdas, dt)
        current = np.array(states, dtype=np.complex128, copy=True)

        for _ in range(steps):
            potential = spectrum.potential(
                self.density(current),
                gravity,
            )

            total_potential = (
                potential + self.physics.drive * drive
            )

            middle = np.empty_like(current)

            for field in range(FIELDS):
                phase = np.ones(DIM, dtype=np.complex128)

                phase[:NUMBERS] = np.exp(
                    -0.5j
                    * dt
                    * total_potential
                    / self.lambdas[field]
                )

                # K * D: prvo potencijalni polukorak,
                # zatim kineticki korak.
                matrix = kinetics[field] * phase[None, :]

                middle[field] = quantum_apply(
                    current[field],
                    matrix,
                    as_circuit=as_circuit,
                )

            updated_potential = spectrum.potential(
                self.density(middle),
                gravity,
            )

            updated_total = (
                updated_potential
                + self.physics.drive * drive
            )

            following = np.empty_like(current)

            for field in range(FIELDS):
                phase = np.ones(DIM, dtype=np.complex128)

                phase[:NUMBERS] = np.exp(
                    -0.5j
                    * dt
                    * updated_total
                    / self.lambdas[field]
                )

                following[field] = quantum_apply(
                    middle[field],
                    np.diag(phase),
                    as_circuit=as_circuit,
                )

            current = following

        norms = np.linalg.norm(current, axis=1)

        require(
            np.all(np.isfinite(current))
            and np.max(np.abs(norms - 1.0)) < 2.0e-8,
            "SP razvoj nije sacuvao normu.",
        )

        require(
            np.max(np.abs(current[:, NUMBERS:])) < 1.0e-12,
            "Masa je presla u nelegalna stanja registra.",
        )

        # Uklanjanje akumulirane greske zaokruzivanja.
        current /= norms[:, None]

        return current


def readout_circuit(theta):
    circuit = QuantumCircuit(2)

    circuit.ry(float(theta[0]), 0)
    circuit.cry(float(theta[1]), 0, 1)
    circuit.cx(1, 0)

    circuit.p(float(theta[2]), 0)
    circuit.p(float(theta[3]), 1)

    return circuit


def readout_state(theta, derivatives=False):
    vector = Statevector.from_instruction(
        readout_circuit(theta)
    ).data

    require(
        abs(vector[3]) < 1.0e-12
        and abs(float(np.vdot(vector, vector).real) - 1.0)
        < 2.0e-12,
        "Neispravno koherentno Qiskit ocitavanje.",
    )

    coefficients = vector[:FIELDS].copy()

    if not derivatives:
        return coefficients

    derivative = np.empty(
        (FIELDS, 4),
        dtype=np.complex128,
    )

    for index in range(4):
        if index < 2:
            shift = np.pi
            denominator = 4.0
        else:
            shift = np.pi / 2.0
            denominator = 2.0

        plus = np.array(theta, dtype=float, copy=True)
        minus = np.array(theta, dtype=float, copy=True)

        plus[index] += shift
        minus[index] -= shift

        derivative[:, index] = (
            readout_state(plus)
            - readout_state(minus)
        ) / denominator

    return coefficients, derivative


def elementary_gram(fields, degree=DRAW_SIZE):
    branches, numbers = fields.shape

    coefficients = np.zeros(
        (degree + 1, branches, branches),
        dtype=np.complex128,
    )

    coefficients[0] = 1.0

    for number in range(numbers):
        product = np.outer(
            fields[:, number].conj(),
            fields[:, number],
        )

        for order in range(
            min(degree, number + 1),
            0,
            -1,
        ):
            coefficients[order] += (
                product * coefficients[order - 1]
            )

    gram = coefficients[degree]

    return 0.5 * (gram + gram.conj().T)


def joint_norm(coefficients, gram):
    value = float(
        np.einsum(
            "r,rs,s->",
            coefficients.conj(),
            gram,
            coefficients,
            optimize=True,
        ).real
    )

    scale = float(np.trace(gram).real)

    if (
        not np.isfinite(value)
        or not np.isfinite(scale)
        or scale <= 0.0
        or value <= MIN_RELATIVE_NORM * scale
    ):
        raise InvalidState(
            "Koherentno stanje je numericki gotovo ponisteno."
        )

    return value


def joint_amplitude(fields, coefficients, combination):
    indices = np.asarray(combination, dtype=np.int64) - 1

    return np.sum(
        coefficients
        * np.prod(fields[:, indices], axis=1)
    )


def joint_marginals(fields, coefficients, degree=DRAW_SIZE):
    branches, numbers = fields.shape

    products = np.einsum(
        "rn,sn->nrs",
        fields.conj(),
        fields,
    )

    prefix = np.zeros(
        (numbers + 1, degree + 1, branches, branches),
        dtype=np.complex128,
    )

    suffix = np.zeros_like(prefix)

    prefix[0, 0] = 1.0
    suffix[numbers, 0] = 1.0

    for number in range(numbers):
        prefix[number + 1] = prefix[number]

        for order in range(1, degree + 1):
            prefix[number + 1, order] += (
                products[number]
                * prefix[number, order - 1]
            )

    for number in range(numbers - 1, -1, -1):
        suffix[number] = suffix[number + 1]

        for order in range(1, degree + 1):
            suffix[number, order] += (
                products[number]
                * suffix[number + 1, order - 1]
            )

    gram = prefix[numbers, degree]
    gram = 0.5 * (gram + gram.conj().T)

    normalizer = joint_norm(coefficients, gram)
    inclusion = np.empty(numbers, dtype=np.float64)

    for number in range(numbers):
        excluded = np.zeros(
            (branches, branches),
            dtype=np.complex128,
        )

        for order in range(degree):
            excluded += (
                prefix[number, order]
                * suffix[
                    number + 1,
                    degree - 1 - order,
                ]
            )

        included_gram = (
            products[number] * excluded
        )

        numerator = np.einsum(
            "r,rs,s->",
            coefficients.conj(),
            included_gram,
            coefficients,
            optimize=True,
        ).real

        inclusion[number] = float(numerator) / normalizer

    require(
        abs(float(np.sum(inclusion)) - degree) < 2.0e-6
        and float(np.min(inclusion)) >= -1.0e-7
        and float(np.max(inclusion)) <= 1.0 + 1.0e-7,
        "Neispravne zajednicke marginale.",
    )

    return inclusion, normalizer


class FieldBank:
    def __init__(
        self,
        draws,
        graph,
        physics,
        announce=False,
    ):
        self.draws = draws
        self.count = len(draws)
        self.physics = physics
        self.evolver = SPEvolver(physics)

        self.forecasts = np.empty(
            (self.count, FIELDS, NUMBERS),
            dtype=np.complex128,
        )

        self.grams = np.empty(
            (self.count, FIELDS, FIELDS),
            dtype=np.complex128,
        )

        current = self.evolver.initial_state()
        zero_drive = np.zeros(NUMBERS, dtype=np.float64)

        for origin, draw in enumerate(draws):
            spectrum = graph.at(origin)

            observed_drive = np.full(
                NUMBERS,
                -1.0,
                dtype=np.float64,
            )

            observed_drive[draw - 1] += (
                NUMBERS / DRAW_SIZE
            )

            # Asimilacija upravo poznatog kola.
            current = self.evolver.advance(
                current,
                observed_drive,
                spectrum,
            )

            # Prognoza nema pristup sledecem ishodu.
            forecast = self.evolver.advance(
                current,
                zero_drive,
                spectrum,
            )

            self.forecasts[origin] = forecast[:, :NUMBERS]
            self.grams[origin] = elementary_gram(
                self.forecasts[origin]
            )

            diagonal = np.diag(
                self.grams[origin]
            ).real

            require(
                np.all(np.isfinite(self.grams[origin]))
                and float(np.max(diagonal)) > 1.0e-260,
                "Polja nemaju stabilnu masu legalnih kombinacija.",
            )

            if announce and (origin + 1) % 1000 == 0:
                print(
                    f"{physics.name} | SP istorija:"
                    f" {origin + 1}/{self.count}"
                )

        self.last_state = current.copy()
        self.last_spectrum = graph.at(self.count - 1)


@dataclass
class FitData:
    amplitudes: np.ndarray
    grams: np.ndarray

    @property
    def count(self):
        return len(self.amplitudes)


def prepare_data(bank, start, end):
    require(
        1 <= start < end <= bank.count,
        "Neispravan interval istorijskih prelaza.",
    )

    targets = np.arange(start, end, dtype=np.int64)
    origins = targets - 1

    amplitudes = np.array(
        [
            np.prod(
                bank.forecasts[target - 1][
                    :,
                    bank.draws[target] - 1,
                ],
                axis=1,
            )
            for target in targets
        ],
        dtype=np.complex128,
    )

    grams = bank.grams[origins].copy()

    scales = np.max(
        np.diagonal(
            grams,
            axis1=1,
            axis2=2,
        ).real,
        axis=1,
    )

    require(
        np.all(np.isfinite(scales))
        and float(np.min(scales)) > 1.0e-260,
        "Neispravna skala kvantnog normalizatora.",
    )

    # Ista skala deli brojilac i imenilac.
    amplitudes /= np.sqrt(scales)[:, None]
    grams /= scales[:, None, None]

    require(
        amplitudes.shape == (len(targets), FIELDS),
        "Neispravne dimenzije amplituda ciljeva.",
    )

    return FitData(amplitudes, grams)


def likelihood(theta, data, gradient=False):
    if gradient:
        coefficients, derivative = readout_state(
            theta,
            derivatives=True,
        )
    else:
        coefficients = readout_state(theta)

    amplitudes = data.amplitudes @ coefficients
    observed_mass = np.abs(amplitudes) ** 2

    normalizers = np.einsum(
        "r,nrs,s->n",
        coefficients.conj(),
        data.grams,
        coefficients,
        optimize=True,
    ).real

    traces = np.trace(
        data.grams,
        axis1=1,
        axis2=2,
    ).real

    if (
        not np.all(np.isfinite(normalizers))
        or np.any(
            normalizers <= MIN_RELATIVE_NORM * traces
        )
        or np.any(observed_mass <= 1.0e-280)
        or not np.all(np.isfinite(observed_mass))
    ):
        raise InvalidState(
            "Numericki nestabilna koherentna verovatnoca."
        )

    probabilities = observed_mass / normalizers

    if float(np.max(probabilities)) > 1.0 + 1.0e-7:
        raise InvalidState(
            "Verovatnoca ishoda prelazi jedan."
        )

    loss = float(
        np.mean(
            np.log(normalizers)
            - np.log(observed_mass)
        )
    )

    if not gradient:
        return loss

    amplitude_derivative = (
        data.amplitudes @ derivative
    )

    normalizer_derivative = 2.0 * np.einsum(
        "r,nrs,sk->nk",
        coefficients.conj(),
        data.grams,
        derivative,
        optimize=True,
    ).real

    observed_derivative = 2.0 * (
        amplitudes.conj()[:, None]
        * amplitude_derivative
    ).real

    grad = np.mean(
        normalizer_derivative / normalizers[:, None]
        - observed_derivative / observed_mass[:, None],
        axis=0,
    )

    if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
        raise InvalidState(
            "NLL ili gradijent nisu konacni."
        )

    return loss, grad


def projected_gradient(theta, gradient):
    projected = gradient.copy()

    for index, (lower, upper) in enumerate(BOUNDS):
        if theta[index] <= lower + 1.0e-9:
            projected[index] = min(
                projected[index],
                0.0,
            )

        if theta[index] >= upper - 1.0e-9:
            projected[index] = max(
                projected[index],
                0.0,
            )

    return float(np.max(np.abs(projected)))


def fit_readout(data, label, warm=None):
    print()
    print(label)
    print(f"Istorijskih prelaza: {data.count}")

    starts = [item.copy() for item in STARTS]

    if warm is not None:
        starts[0] = warm.copy()

    overall = None

    for start_number, start in enumerate(starts, start=1):
        best = {
            "loss": math.inf,
            "theta": None,
            "gradient": None,
        }

        calls = 0

        def objective(theta):
            nonlocal calls
            calls += 1

            try:
                value, grad = likelihood(
                    theta,
                    data,
                    gradient=True,
                )
            except InvalidState:
                return 1.0e100, np.zeros(4)

            if value < best["loss"]:
                best["loss"] = float(value)
                best["theta"] = theta.copy()
                best["gradient"] = grad.copy()

            return value, grad

        initial_value, _ = objective(start)

        if not np.isfinite(initial_value) or initial_value >= 1.0e99:
            print(
                f"Pocetak {start_number}:"
                " numericki neupotrebljivo stanje."
            )
            continue

        optimized = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=BOUNDS,
            options={
                "maxiter": MAX_ITERATIONS,
                "maxfun": MAX_EVALUATIONS,
                "maxls": 20,
                "ftol": 1.0e-11,
                "gtol": GRADIENT_TOLERANCE,
            },
        )

        require(
            best["theta"] is not None,
            "Izgubljen pocetni konacni kandidat.",
        )

        same_candidate = bool(
            np.max(
                np.abs(best["theta"] - optimized.x)
            ) < 1.0e-10
            and np.isfinite(optimized.fun)
            and optimized.fun < 1.0e99
        )

        candidate = {
            "theta": best["theta"].copy(),
            "loss": best["loss"],
            "gradient": float(
                np.max(np.abs(best["gradient"]))
            ),
            "projected_gradient": projected_gradient(
                best["theta"],
                best["gradient"],
            ),
            "converged": bool(
                optimized.success and same_candidate
            ),
        }

        print(
            f"Pocetak {start_number}"
            f" | NLL: {candidate['loss']:.9f}"
            f" | poziva: {calls}"
            f" | projektovani grad:"
            f" {candidate['projected_gradient']:.3e}"
        )
        print(f"Status: {optimized.message}")

        if (
            overall is None
            or candidate["loss"] < overall["loss"]
        ):
            overall = candidate

    require(
        overall is not None,
        "Nijedan pocetak nije dao konacan model.",
    )

    if not overall["converged"]:
        print(
            "DIJAGNOSTIKA: konvergencija izabranog"
            " kandidata nije potvrdjena."
        )

    return overall


def exhaustive_next(
    fields,
    coefficients,
    degree=DRAW_SIZE,
    announce=True,
):
    numbers = fields.shape[1]
    gram = elementary_gram(fields, degree)
    normalizer = joint_norm(coefficients, gram)

    expected_count = math.comb(numbers, degree)

    iterator = itertools.combinations(
        range(numbers),
        degree,
    )

    best_mass = -1.0
    best_combination = None

    total_mass = 0.0
    compensation = 0.0
    count = 0
    next_report = 2_000_000

    started = time.monotonic()

    while True:
        items = list(
            itertools.islice(iterator, SEARCH_CHUNK)
        )

        if not items:
            break

        block = np.asarray(items, dtype=np.int16)

        branch_amplitudes = np.prod(
            fields[:, block],
            axis=2,
        )

        amplitudes = np.sum(
            coefficients[:, None]
            * branch_amplitudes,
            axis=0,
        )

        masses = np.abs(amplitudes) ** 2
        index = int(np.argmax(masses))

        if float(masses[index]) > best_mass:
            best_mass = float(masses[index])
            best_combination = block[index].astype(
                np.int64
            ) + 1

        addition = float(np.sum(masses)) - compensation
        updated = total_mass + addition
        compensation = (
            updated - total_mass
        ) - addition
        total_mass = updated

        count += len(block)

        if announce and count >= next_report:
            print(
                f"NEXT pretraga: {count:,}/{expected_count:,}"
                f" | proteklo:"
                f" {time.monotonic() - started:.1f} s"
            )

            next_report += 2_000_000

    require(
        count == expected_count
        and best_combination is not None
        and best_mass > 0.0,
        "Iscrpna pretraga nije zavrsena.",
    )

    total_probability = total_mass / normalizer

    require(
        abs(total_probability - 1.0) < 1.0e-6,
        "Iscrpni zbir i analiticki normalizator se razlikuju.",
    )

    require(
        len(set(best_combination.tolist())) == degree
        and best_combination[0] >= 1
        and best_combination[-1] <= numbers,
        "NEXT nije legalna kombinacija.",
    )

    return {
        "next": best_combination,
        "probability": best_mass / normalizer,
        "total": total_probability,
        "normalizer": normalizer,
        "count": count,
    }


def qiskit_combination_amplitude(fields, theta, combination):
    degree = len(combination)
    circuit = QuantumCircuit(degree + 2)

    circuit.compose(
        readout_circuit(theta),
        qubits=[0, 1],
        inplace=True,
    )

    for slot, value in enumerate(combination):
        for branch in range(FIELDS):
            alpha = complex(
                fields[branch, int(value) - 1]
            )

            mass = abs(alpha) ** 2

            require(
                mass <= 1.0 + 1.0e-12,
                "Amplituda pojedinacnog registra prelazi jedan.",
            )

            beta = math.sqrt(
                max(0.0, 1.0 - mass)
            )

            flag_unitary = np.array(
                [
                    [beta, -alpha.conjugate()],
                    [alpha, beta],
                ],
                dtype=np.complex128,
            )

            controlled = np.eye(
                8,
                dtype=np.complex128,
            )

            selected = [branch, branch + 4]

            controlled[
                np.ix_(selected, selected)
            ] = flag_unitary

            circuit.unitary(
                controlled,
                [0, 1, slot + 2],
            )

    circuit.h(0)
    circuit.h(1)

    output = Statevector.from_instruction(circuit).data

    all_flags_one = ((1 << degree) - 1) << 2

    # Svaka od tri grane ima preklapanje 1/2 sa ancila |++>.
    return 2.0 * output[all_flags_one]


def verify_amplitude(fields, theta, combination):
    coefficients = readout_state(theta)

    compact = joint_amplitude(
        fields,
        coefficients,
        combination,
    )

    explicit = qiskit_combination_amplitude(
        fields,
        theta,
        combination,
    )

    error = abs(explicit - compact)

    require(
        error < 2.0e-11,
        "Qiskit interferencija ne odgovara zajednickoj amplitudi.",
    )

    if abs(compact) > 1.0e-12:
        require(
            error / abs(compact) < 1.0e-8,
            "Relativna greska zajednicke amplitude je previsoka.",
        )

    return float(error)


def deterministic_draws(count):
    offsets = np.array([0, 3, 8, 14, 21, 25, 33])

    return np.array(
        [
            np.sort(
                (offsets + 5 * row + row * row)
                % NUMBERS + 1
            )
            for row in range(count)
        ],
        dtype=np.int64,
    )


def self_checks():
    print()
    print("Pokrecem provere pre treninga.")

    draws = deterministic_draws(13)
    graph = GraphHistory(draws)

    spectrum = graph.at(8)
    model = SPEvolver(PHYSICS[1])

    positions = np.arange(1, NUMBERS + 1, dtype=float)

    states = np.zeros(
        (FIELDS, DIM),
        dtype=np.complex128,
    )

    for field in range(FIELDS):
        row = (
            1.0
            + 0.2 * np.cos((field + 1) * positions)
        ) * np.exp(
            0.17j * (field + 1) * positions
        )

        states[field, :NUMBERS] = row / np.linalg.norm(row)

    density = model.density(states)
    potential = spectrum.potential(
        density,
        model.physics.gravity,
    )

    contrast = NUMBERS * (
        density - float(np.mean(density))
    )

    poisson_error = float(
        np.max(
            np.abs(
                -spectrum.laplacian(potential)
                - model.physics.gravity * contrast
            )
        )
    )

    require(
        poisson_error < 2.0e-10,
        "Poissonova jednacina nije prosla proveru.",
    )

    drive = np.full(NUMBERS, -1.0)
    drive[draws[8] - 1] += NUMBERS / DRAW_SIZE

    evolved = model.advance(
        states,
        drive,
        spectrum,
        duration=0.25,
    )

    reversed_state = model.advance(
        evolved,
        drive,
        spectrum,
        duration=-0.25,
    )

    reversal_error = float(
        np.max(np.abs(reversed_state - states))
    )

    require(
        reversal_error < 2.0e-10,
        "Simetricni SP korak nije vremenski reverzibilan.",
    )

    circuit_evolved = model.advance(
        states,
        drive,
        spectrum,
        duration=0.25,
        as_circuit=True,
    )

    circuit_error = float(
        np.max(np.abs(circuit_evolved - evolved))
    )

    require(
        circuit_error < 2.0e-10,
        "Qiskit kola i operatori daju razlicit SP razvoj.",
    )

    small_numbers = 8
    degree = 3
    small_fields = np.empty(
        (FIELDS, small_numbers),
        dtype=np.complex128,
    )

    index = np.arange(1, small_numbers + 1, dtype=float)

    for field in range(FIELDS):
        row = (
            1.0
            + 0.3 * np.cos((field + 1) * index)
        ) * np.exp(
            0.27j * (field + 1) * index
        )

        small_fields[field] = row / np.linalg.norm(row)

    theta = np.array([1.1, 1.4, 0.3, -0.4])
    coefficients = readout_state(theta)

    searched = exhaustive_next(
        small_fields,
        coefficients,
        degree=degree,
        announce=False,
    )

    inclusion, normalizer = joint_marginals(
        small_fields,
        coefficients,
        degree=degree,
    )

    brute_inclusion = np.zeros(small_numbers)
    brute_total = 0.0

    for combination in itertools.combinations(
        range(1, small_numbers + 1),
        degree,
    ):
        amplitude = joint_amplitude(
            small_fields,
            coefficients,
            combination,
        )

        probability = (
            abs(amplitude) ** 2 / normalizer
        )

        brute_total += probability
        brute_inclusion[
            np.asarray(combination) - 1
        ] += probability

    require(
        abs(brute_total - 1.0) < 2.0e-11
        and np.max(
            np.abs(brute_inclusion - inclusion)
        ) < 2.0e-11,
        "Normalizator ili marginale nisu prosli iscrpnu proveru.",
    )

    qiskit_error = verify_amplitude(
        small_fields,
        theta,
        searched["next"],
    )

    combinations = list(
        itertools.combinations(
            range(1, small_numbers + 1),
            degree,
        )
    )[::7]

    amplitudes = np.array(
        [
            np.prod(
                small_fields[
                    :,
                    np.asarray(combination) - 1,
                ],
                axis=1,
            )
            for combination in combinations
        ]
    )

    grams = np.repeat(
        elementary_gram(
            small_fields,
            degree,
        )[None, :, :],
        len(amplitudes),
        axis=0,
    )

    scale = float(
        np.max(np.diag(grams[0]).real)
    )

    data = FitData(
        amplitudes / np.sqrt(scale),
        grams / scale,
    )

    _, analytic = likelihood(
        theta,
        data,
        gradient=True,
    )

    largest_gradient_error = 0.0
    step = 1.0e-6

    for parameter in range(4):
        plus = theta.copy()
        minus = theta.copy()

        plus[parameter] += step
        minus[parameter] -= step

        numeric = (
            likelihood(plus, data)
            - likelihood(minus, data)
        ) / (2.0 * step)

        largest_gradient_error = max(
            largest_gradient_error,
            abs(numeric - analytic[parameter]),
        )

    require(
        largest_gradient_error < 3.0e-6,
        "Gradijent kvantnog ocitavanja nije prosao proveru.",
    )

    # Jednake amplitude moraju dati jednaku verovatnocu
    # svim legalnim kombinacijama u ovom granicnom slucaju.
    flat_fields = np.full(
        (FIELDS, NUMBERS),
        1.0 / np.sqrt(NUMBERS),
        dtype=np.complex128,
    )

    flat_normalizer = joint_norm(
        coefficients,
        elementary_gram(flat_fields),
    )

    expected = 1.0 / math.comb(NUMBERS, DRAW_SIZE)

    for combination in (
        np.arange(1, 8),
        np.arange(33, 40),
        np.array([2, 8, 13, 19, 25, 31, 38]),
    ):
        probability = abs(
            joint_amplitude(
                flat_fields,
                coefficients,
                combination,
            )
        ) ** 2 / flat_normalizer

        require(
            abs(probability / expected - 1.0) < 2.0e-10,
            "Granicni slucaj ima vestacku prednost krajnjeg niza.",
        )

    short_graph = GraphHistory(draws[:9])

    short_bank = FieldBank(
        draws[:9],
        short_graph,
        PHYSICS[0],
    )

    long_bank = FieldBank(
        draws,
        graph,
        PHYSICS[0],
    )

    causal_error = float(
        np.max(
            np.abs(
                short_bank.forecasts
                - long_bank.forecasts[:9]
            )
        )
    )

    require(
        causal_error < 2.0e-10,
        "Buduca kola menjaju ranije prognozne amplitude.",
    )

    print(f"Poissonov rezidual: {poisson_error:.3e}")
    print(f"Greska vremenske reverzibilnosti: {reversal_error:.3e}")
    print(f"Greska Qiskit kola/operatora: {circuit_error:.3e}")
    print(f"Greska koherentne amplitude: {qiskit_error:.3e}")
    print(f"Greska gradijenta: {largest_gradient_error:.3e}")
    print(f"Greska kauzalnosti: {causal_error:.3e}")
    print("Iscrpna normalizacija i marginale: prosle.")
    print("Granicni slucaj bez prednosti krajnjeg niza: prosao.")
    print("Ugradjene provere su prosle.")


def read_draws(path):
    require(path.is_file(), f"CSV nije pronadjen: {path}")

    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    lines = raw.decode("utf-8-sig").splitlines()

    first = next(
        (line for line in lines if line.strip()),
        None,
    )

    require(first is not None, f"CSV je prazan: {path.name}")

    if "," in first:
        reader = csv.reader(lines, delimiter=",")
    elif ";" in first:
        reader = csv.reader(lines, delimiter=";")
    else:
        reader = (line.split() for line in lines)

    draws = []
    blank = 0

    for line_number, row in enumerate(reader, start=1):
        if not row or all(not cell.strip() for cell in row):
            blank += 1
            continue

        require(
            len(row) == DRAW_SIZE,
            f"{path.name}, red {line_number}:"
            " ocekivano je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as error:
            raise RuntimeError(
                f"{path.name}, red {line_number}:"
                " ocekivani su celi brojevi bez zaglavlja."
            ) from error

        require(
            len(set(values)) == DRAW_SIZE
            and min(values) >= 1
            and max(values) <= NUMBERS,
            f"{path.name}, red {line_number}:"
            " nelegalna kombinacija.",
        )

        draws.append(sorted(values))

    array = np.asarray(draws, dtype=np.int64)

    require(
        len(array) >= 100,
        f"{path.name}: premalo kola za zadatu validaciju.",
    )

    print()
    print(f"ULAZ: {path.name}")
    print(f"Redova: {len(array)} | praznih redova: {blank}")
    print(f"SHA256: {fingerprint}")
    print("Redosled kola ostaje redosled iz CSV-a.")

    return array, fingerprint


def longest_run(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_file(path, draws, fingerprint):
    print()
    print(f"OBRADA: {path.name}")

    graph = GraphHistory(draws, announce=True)
    count = len(draws)

    cut60 = int(0.60 * count)
    cut70 = int(0.70 * count)
    cut80 = int(0.80 * count)

    candidates = []

    for physics in PHYSICS:
        print()
        print(
            f"{physics.name}"
            f" | lambda: {physics.wavelength:g}"
            f" | gravitaciona veza: {physics.gravity:g}"
            f" | spoljna veza: {physics.drive:g}"
        )

        bank = FieldBank(
            draws,
            graph,
            physics,
            announce=True,
        )

        score_sum = 0.0
        score_count = 0
        warm = None

        for fold_number, (train_end, validation_end) in enumerate(
            (
                (cut60, cut70),
                (cut70, cut80),
            ),
            start=1,
        ):
            training = prepare_data(
                bank,
                1,
                train_end,
            )

            fitted = fit_readout(
                training,
                (
                    f"{path.name}"
                    f" | {physics.name}"
                    f" | validacija {fold_number}"
                ),
                warm=warm,
            )

            validation = prepare_data(
                bank,
                train_end,
                validation_end,
            )

            score = likelihood(
                fitted["theta"],
                validation,
            )

            print(f"Validacioni NLL: {score:.9f}")

            score_sum += score * validation.count
            score_count += validation.count
            warm = fitted["theta"].copy()

        candidates.append(
            {
                "bank": bank,
                "score": score_sum / score_count,
                "theta": warm,
            }
        )

    selected = min(
        candidates,
        key=lambda item: item["score"],
    )

    bank = selected["bank"]

    print()
    print(f"Izabrana dinamika: {bank.physics.name}")
    print(f"Objedinjeni validacioni NLL: {selected['score']:.9f}")

    training = prepare_data(
        bank,
        1,
        cut80,
    )

    test_fit = fit_readout(
        training,
        f"{path.name} | trening pre odvojenog testa",
        warm=selected["theta"],
    )

    testing = prepare_data(
        bank,
        cut80,
        count,
    )

    test_nll = likelihood(
        test_fit["theta"],
        testing,
    )

    print()
    print(f"Odvojeni test NLL: {test_nll:.9f}")
    print(
        "Test je sekvencijalan."
        " Koeficijenti ostaju fiksni,"
        " a polja vide samo prethodno ostvarena kola."
    )

    full_training = prepare_data(
        bank,
        1,
        count,
    )

    final_fit = fit_readout(
        full_training,
        f"{path.name} | zavrsno ucenje na celoj istoriji",
        warm=test_fit["theta"],
    )

    theta = final_fit["theta"]
    coefficients = readout_state(theta)
    fields = bank.forecasts[-1]

    inclusion, _ = joint_marginals(
        fields,
        coefficients,
    )

    zero_drive = np.zeros(NUMBERS)

    no_self_potential = bank.evolver.advance(
        bank.last_state,
        zero_drive,
        bank.last_spectrum,
        gravity=0.0,
    )[:, :NUMBERS]

    no_potential_inclusion, _ = joint_marginals(
        no_self_potential,
        coefficients,
    )

    potential_effect = float(
        np.max(
            np.abs(
                inclusion - no_potential_inclusion
            )
        )
    )

    earlier = max(0, count - 12)

    earlier_inclusion, _ = joint_marginals(
        bank.forecasts[earlier],
        coefficients,
    )

    history_effect = float(
        np.max(
            np.abs(
                inclusion - earlier_inclusion
            )
        )
    )

    refined = bank.evolver.advance(
        bank.last_state,
        zero_drive,
        bank.last_spectrum,
        steps=2 * SUBSTEPS,
    )[:, :NUMBERS]

    refined_inclusion, _ = joint_marginals(
        refined,
        coefficients,
    )

    refinement_effect = float(
        np.max(
            np.abs(
                inclusion - refined_inclusion
            )
        )
    )

    print()
    print("Pocinjem iscrpni racun zajednickog NEXT maksimuma.")

    result = exhaustive_next(
        fields,
        coefficients,
    )

    qiskit_error = verify_amplitude(
        fields,
        theta,
        result["next"],
    )

    print()
    print(path.name)
    print(f"NEXT: {number_text(result['next'])}")
    print(f"Dinamika: {bank.physics.name}")
    print(f"Validacioni NLL: {selected['score']:.9f}")
    print(f"Odvojeni test NLL: {test_nll:.9f}")
    print(f"Modelna verovatnoca NEXT: {result['probability']:.12g}")
    print(f"Zbir svih verovatnoca: {result['total']:.12f}")
    print(f"Pregledanih kombinacija: {result['count']:,}")
    print(f"Greska Qiskit zajednicke amplitude: {qiskit_error:.3e}")
    print(
        f"Efekat samopotencijala u narednom koraku:"
        f" {potential_effect:.6e}"
    )
    print(
        f"Promena marginala prema ranijem kontekstu:"
        f" {history_effect:.6e}"
    )
    print(
        f"Osetljivost narednog koraka na finiju integraciju:"
        f" {refinement_effect:.6e}"
    )

    for branch, coefficient in enumerate(coefficients, start=1):
        print(
            f"Koherentni koeficijent {branch}:"
            f" {coefficient.real:.6f}"
            f"{coefficient.imag:+.6f}j"
        )

    for lag in range(1, 4):
        overlap = len(
            set(result["next"].tolist())
            & set(draws[-lag].tolist())
        )

        print(
            f"Presek sa kolom unazad {lag}: {overlap}/7"
        )

        if overlap >= 6:
            print(
                "DIJAGNOSTIKA: veliko preklapanje"
                " sa istorijskim kolom."
            )

    run = longest_run(result["next"])

    if run >= 5:
        print(
            f"DIJAGNOSTIKA: niz od {run}"
            " uzastopnih brojeva."
        )

    if int(result["next"][-1] - result["next"][0]) <= 9:
        print(
            "DIJAGNOSTIKA: svih sedam brojeva"
            " nalazi se u rasponu od deset brojeva."
        )

    if potential_effect < 1.0e-6:
        print(
            "DIJAGNOSTIKA: slab efekat samopotencijala"
            " u narednom koraku."
        )

    if history_effect < 1.0e-6:
        print(
            "DIJAGNOSTIKA: mala promena raspodele"
            " izmedju pregledanih konteksta."
        )

    if refinement_effect > 1.0e-3:
        print(
            "DIJAGNOSTIKA: primetna osetljivost"
            " na vremensku diskretizaciju."
        )

    if not final_fit["converged"]:
        print(
            "DIJAGNOSTIKA: konvergencija zavrsnog"
            " kandidata nije potvrdjena."
        )

    print(
        "Dijagnostika ne menja NEXT"
        " i ne potvrdjuje prediktivni kvalitet."
    )

    return {
        "path": path,
        "fingerprint": fingerprint,
        "next": result["next"].copy(),
        "physics": bank.physics.name,
        "test_nll": test_nll,
        "inclusion": inclusion.copy(),
        "potential_effect": potential_effect,
        "history_effect": history_effect,
        "converged": final_fit["converged"],
        "gradient": final_fit["gradient"],
        "projected_gradient": final_fit["projected_gradient"],
    }


def main():
    print("QUANTUM DARK MATTER SP V1 — Loto 7/39")
    print(f"Python: {platform.python_version()}")
    print(f"Qiskit: {qiskit.__version__}")
    print("Tri povezana Schrodinger-Poisson polja.")
    print("Kauzalni graf 39 simbola.")
    print("Gustina polja odredjuje sopstveni potencijal.")
    print("Potencijal menja narednu Qiskit evoluciju.")
    print("Sedam sestokubitnih registara u kompaktnoj reprezentaciji.")
    print("Koherentna amplituda cele kombinacije.")
    print("Regresija raspodele zajednickim NLL kriterijumom.")
    print("Parametri dinamike biraju se hronoloskom validacijom.")
    print("Globalni NEXT maksimum dobija se iscrpnom pretragom.")
    print("Lokalna optimizacija uglova ne garantuje globalni optimum.")
    print("Bez uzorkovanja, slucajnih kombinacija i zapisivanja fajlova.")

    loaded = []

    for path in CSV_PATHS:
        draws, fingerprint = read_draws(path)
        loaded.append((path, draws, fingerprint))

    self_checks()

    results = []

    for path, draws, fingerprint in loaded:
        results.append(
            process_file(
                path,
                draws,
                fingerprint,
            )
        )

    print()
    print("ZAVRSNI NEXT REZULTATI")

    for result in results:
        print(
            f"{result['path'].name}"
            f" | NEXT: {number_text(result['next'])}"
            f" | dinamika: {result['physics']}"
            f" | test NLL: {result['test_nll']:.9f}"
        )

        status = (
            "optimizator prijavio konvergenciju izabranog kandidata"
            if result["converged"]
            else "konvergencija izabranog kandidata nije potvrdjena"
        )

        print(
            f"Status: {status}"
            f" | Max|grad|: {result['gradient']:.6e}"
            f" | Max|projektovani grad|:"
            f" {result['projected_gradient']:.6e}"
        )

    for first, second in itertools.combinations(results, 2):
        if np.array_equal(first["next"], second["next"]):
            difference = float(
                np.max(
                    np.abs(
                        first["inclusion"]
                        - second["inclusion"]
                    )
                )
            )

            print()
            print(
                f"DIJAGNOSTIKA: isti NEXT za"
                f" {first['path'].name}"
                f" i {second['path'].name}."
            )
            print(
                f"Najveca razlika marginalnih raspodela:"
                f" {difference:.6e}"
            )
            print(
                "Rezultati nisu menjani"
                " radi vestackog razlikovanja."
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print(f"ZAUSTAVLJENO: {error}")
        print("Zavrsni skup od tri NEXT rezultata nije potvrdjen.")
        raise



"""
QUANTUM DARK MATTER SP V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Tri povezana Schrodinger-Poisson polja.
Kauzalni graf 39 simbola.
Gustina polja odredjuje sopstveni potencijal.
Potencijal menja narednu Qiskit evoluciju.
Sedam sestokubitnih registara u kompaktnoj reprezentaciji.
Koherentna amplituda cele kombinacije.
Regresija raspodele zajednickim NLL kriterijumom.
Parametri dinamike biraju se hronoloskom validacijom.
Globalni NEXT maksimum dobija se iscrpnom pretragom.
Lokalna optimizacija uglova ne garantuje globalni optimum.
Bez uzorkovanja, slucajnih kombinacija i zapisivanja fajlova.

ULAZ: loto7_4712_k87.csv
Redova: 4712 | praznih redova: 0
SHA256: d5504ffa43c8d7ce477ec31e3a0fff18e06ddae6a9fda656fda8813439750808
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4712_k87_loto_2978.csv
Redova: 2978 | praznih redova: 0
SHA256: 0b1abfe53d71a4c52a467be9254d094b3e48113ae875c2a154dc6229e763d1a2
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4712_k87_loto_plus_1734.csv
Redova: 1734 | praznih redova: 0
SHA256: f2fd75c1e476476a5691c74e942cfee2e4628ee0662c68433f46a7d9efd32b9e
Redosled kola ostaje redosled iz CSV-a.

Pokrecem provere pre treninga.
Poissonov rezidual: 1.128e-17
Greska vremenske reverzibilnosti: 3.455e-15
Greska Qiskit kola/operatora: 8.442e-17
Greska koherentne amplitude: 1.552e-17
Greska gradijenta: 4.311e-10
Greska kauzalnosti: 1.717e-16
Iscrpna normalizacija i marginale: prosle.
Granicni slucaj bez prednosti krajnjeg niza: prosao.
Ugradjene provere su prosle.

OBRADA: loto7_4712_k87.csv
Kauzalni graf: 1000/4712
Kauzalni graf: 2000/4712
Kauzalni graf: 3000/4712
Kauzalni graf: 4000/4712

SP_A | lambda: 0.5 | gravitaciona veza: 0.005 | spoljna veza: 0.1
SP_A | SP istorija: 1000/4712
SP_A | SP istorija: 2000/4712
SP_A | SP istorija: 3000/4712
SP_A | SP istorija: 4000/4712

loto7_4712_k87.csv | SP_A | validacija 1
Istorijskih prelaza: 2826
Pocetak 1 | NLL: 18.325009337 | poziva: 35 | projektovani grad: 9.548e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.303543994 | poziva: 21 | projektovani grad: 2.920e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.338785441 | poziva: 14 | projektovani grad: 1.775e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.369635253

loto7_4712_k87.csv | SP_A | validacija 2
Istorijskih prelaza: 3297
Pocetak 1 | NLL: 18.312399809 | poziva: 9 | projektovani grad: 7.872e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.308839958 | poziva: 31 | projektovani grad: 4.008e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.345927679 | poziva: 14 | projektovani grad: 2.613e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.540676251

SP_B | lambda: 0.9 | gravitaciona veza: 0.01 | spoljna veza: 0.18
SP_B | SP istorija: 1000/4712
SP_B | SP istorija: 2000/4712
SP_B | SP istorija: 3000/4712
SP_B | SP istorija: 4000/4712

loto7_4712_k87.csv | SP_B | validacija 1
Istorijskih prelaza: 2826
Pocetak 1 | NLL: 18.461681314 | poziva: 25 | projektovani grad: 9.377e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.423540163 | poziva: 21 | projektovani grad: 7.238e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.447144902 | poziva: 23 | projektovani grad: 7.743e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.928033981

loto7_4712_k87.csv | SP_B | validacija 2
Istorijskih prelaza: 3297
Pocetak 1 | NLL: 18.495538851 | poziva: 9 | projektovani grad: 1.938e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.495538851 | poziva: 26 | projektovani grad: 4.808e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.513734147 | poziva: 12 | projektovani grad: 6.438e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.685555991

SP_C | lambda: 1.3 | gravitaciona veza: 0.02 | spoljna veza: 0.26
SP_C | SP istorija: 1000/4712
SP_C | SP istorija: 2000/4712
SP_C | SP istorija: 3000/4712
SP_C | SP istorija: 4000/4712

loto7_4712_k87.csv | SP_C | validacija 1
Istorijskih prelaza: 2826
Pocetak 1 | NLL: 18.358420209 | poziva: 15 | projektovani grad: 1.015e-06
Status: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
Pocetak 2 | NLL: 18.372539914 | poziva: 14 | projektovani grad: 4.583e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.355504159 | poziva: 14 | projektovani grad: 9.730e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.328416287

loto7_4712_k87.csv | SP_C | validacija 2
Istorijskih prelaza: 3297
Pocetak 1 | NLL: 18.351186610 | poziva: 12 | projektovani grad: 2.246e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.375975677 | poziva: 13 | projektovani grad: 1.310e-06
Status: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
Pocetak 3 | NLL: 18.351186610 | poziva: 25 | projektovani grad: 6.400e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.216468772

Izabrana dinamika: SP_C
Objedinjeni validacioni NLL: 18.272442530

loto7_4712_k87.csv | trening pre odvojenog testa
Istorijskih prelaza: 3768
Pocetak 1 | NLL: 18.334244617 | poziva: 11 | projektovani grad: 1.826e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.358918220 | poziva: 14 | projektovani grad: 6.280e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.334244617 | poziva: 24 | projektovani grad: 1.572e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Odvojeni test NLL: 18.510931827
Test je sekvencijalan. Koeficijenti ostaju fiksni, a polja vide samo prethodno ostvarena kola.

loto7_4712_k87.csv | zavrsno ucenje na celoj istoriji
Istorijskih prelaza: 4711
Pocetak 1 | NLL: 18.369590738 | poziva: 11 | projektovani grad: 8.883e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.395898916 | poziva: 16 | projektovani grad: 1.773e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.369590738 | poziva: 17 | projektovani grad: 3.386e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Pocinjem iscrpni racun zajednickog NEXT maksimuma.
NEXT pretraga: 2,031,616/15,380,937 | proteklo: 0.9 s
NEXT pretraga: 4,063,232/15,380,937 | proteklo: 1.8 s
NEXT pretraga: 6,029,312/15,380,937 | proteklo: 2.6 s
NEXT pretraga: 8,060,928/15,380,937 | proteklo: 3.5 s
NEXT pretraga: 10,027,008/15,380,937 | proteklo: 4.3 s
NEXT pretraga: 12,058,624/15,380,937 | proteklo: 5.2 s
NEXT pretraga: 14,024,704/15,380,937 | proteklo: 6.0 s

loto7_4712_k87.csv
NEXT: 2 x 14 y 19 z 34
Dinamika: SP_C
Validacioni NLL: 18.272442530
Odvojeni test NLL: 18.510931827
Modelna verovatnoca NEXT: 3.89480535616e-05
Zbir svih verovatnoca: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greska Qiskit zajednicke amplitude: 6.776e-21
Efekat samopotencijala u narednom koraku: 1.825549e-04
Promena marginala prema ranijem kontekstu: 1.461696e-01
Osetljivost narednog koraka na finiju integraciju: 1.558212e-06
Koherentni koeficijent 1: 0.543481+0.000000j
Koherentni koeficijent 2: 0.341185-0.462234j
Koherentni koeficijent 3: 0.570022+0.222791j
Presek sa kolom unazad 1: 0/7
Presek sa kolom unazad 2: 2/7
Presek sa kolom unazad 3: 3/7
Dijagnostika ne menja NEXT i ne potvrdjuje prediktivni kvalitet.

OBRADA: loto7_4712_k87_loto_2978.csv
Kauzalni graf: 1000/2978
Kauzalni graf: 2000/2978

SP_A | lambda: 0.5 | gravitaciona veza: 0.005 | spoljna veza: 0.1
SP_A | SP istorija: 1000/2978
SP_A | SP istorija: 2000/2978

loto7_4712_k87_loto_2978.csv | SP_A | validacija 1
Istorijskih prelaza: 1785
Pocetak 1 | NLL: 18.331207432 | poziva: 28 | projektovani grad: 1.508e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.321341430 | poziva: 28 | projektovani grad: 4.801e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.325326687 | poziva: 21 | projektovani grad: 1.047e-06
Status: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
Validacioni NLL: 18.326621269

loto7_4712_k87_loto_2978.csv | SP_A | validacija 2
Istorijskih prelaza: 2083
Pocetak 1 | NLL: 18.320431527 | poziva: 11 | projektovani grad: 3.569e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.322596532 | poziva: 16 | projektovani grad: 6.532e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.305065635 | poziva: 32 | projektovani grad: 5.437e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.432323415

SP_B | lambda: 0.9 | gravitaciona veza: 0.01 | spoljna veza: 0.18
SP_B | SP istorija: 1000/2978
SP_B | SP istorija: 2000/2978

loto7_4712_k87_loto_2978.csv | SP_B | validacija 1
Istorijskih prelaza: 1785
Pocetak 1 | NLL: 18.346639543 | poziva: 28 | projektovani grad: 1.427e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.341268957 | poziva: 24 | projektovani grad: 9.041e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.413032500 | poziva: 34 | projektovani grad: 9.278e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.461221826

loto7_4712_k87_loto_2978.csv | SP_B | validacija 2
Istorijskih prelaza: 2083
Pocetak 1 | NLL: 18.357703275 | poziva: 12 | projektovani grad: 3.709e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.357703275 | poziva: 19 | projektovani grad: 6.625e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.383989068 | poziva: 32 | projektovani grad: 1.294e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.521236157

SP_C | lambda: 1.3 | gravitaciona veza: 0.02 | spoljna veza: 0.26
SP_C | SP istorija: 1000/2978
SP_C | SP istorija: 2000/2978

loto7_4712_k87_loto_2978.csv | SP_C | validacija 1
Istorijskih prelaza: 1785
Pocetak 1 | NLL: 18.430658302 | poziva: 16 | projektovani grad: 1.582e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.420726618 | poziva: 27 | projektovani grad: 5.515e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.399828856 | poziva: 37 | projektovani grad: 9.664e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.327586979

loto7_4712_k87_loto_2978.csv | SP_C | validacija 2
Istorijskih prelaza: 2083
Pocetak 1 | NLL: 18.388889790 | poziva: 11 | projektovani grad: 5.554e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.397067170 | poziva: 19 | projektovani grad: 2.696e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.399657563 | poziva: 28 | projektovani grad: 3.643e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.263909490

Izabrana dinamika: SP_C
Objedinjeni validacioni NLL: 18.295748235

loto7_4712_k87_loto_2978.csv | trening pre odvojenog testa
Istorijskih prelaza: 2381
Pocetak 1 | NLL: 18.372124287 | poziva: 14 | projektovani grad: 2.095e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.362529760 | poziva: 15 | projektovani grad: 5.275e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.389804211 | poziva: 26 | projektovani grad: 2.072e-06
Status: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH

Odvojeni test NLL: 18.476708429
Test je sekvencijalan. Koeficijenti ostaju fiksni, a polja vide samo prethodno ostvarena kola.

loto7_4712_k87_loto_2978.csv | zavrsno ucenje na celoj istoriji
Istorijskih prelaza: 2977
Pocetak 1 | NLL: 18.385259908 | poziva: 9 | projektovani grad: 9.230e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.385604741 | poziva: 23 | projektovani grad: 5.964e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.412235116 | poziva: 30 | projektovani grad: 3.377e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Pocinjem iscrpni racun zajednickog NEXT maksimuma.
NEXT pretraga: 2,031,616/15,380,937 | proteklo: 0.9 s
NEXT pretraga: 4,063,232/15,380,937 | proteklo: 1.8 s
NEXT pretraga: 6,029,312/15,380,937 | proteklo: 2.6 s
NEXT pretraga: 8,060,928/15,380,937 | proteklo: 3.5 s
NEXT pretraga: 10,027,008/15,380,937 | proteklo: 4.3 s
NEXT pretraga: 12,058,624/15,380,937 | proteklo: 5.2 s
NEXT pretraga: 14,024,704/15,380,937 | proteklo: 6.1 s

loto7_4712_k87_loto_2978.csv
NEXT: 8 x 19 y 29 z 34
Dinamika: SP_C
Validacioni NLL: 18.295748235
Odvojeni test NLL: 18.476708429
Modelna verovatnoca NEXT: 2.33526954661e-05
Zbir svih verovatnoca: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greska Qiskit zajednicke amplitude: 1.366e-20
Efekat samopotencijala u narednom koraku: 2.178962e-04
Promena marginala prema ranijem kontekstu: 9.487366e-02
Osetljivost narednog koraka na finiju integraciju: 2.751230e-06
Koherentni koeficijent 1: 0.567043+0.000000j
Koherentni koeficijent 2: 0.325458+0.489022j
Koherentni koeficijent 3: 0.564944-0.119315j
Presek sa kolom unazad 1: 3/7
Presek sa kolom unazad 2: 1/7
Presek sa kolom unazad 3: 1/7
Dijagnostika ne menja NEXT i ne potvrdjuje prediktivni kvalitet.

OBRADA: loto7_4712_k87_loto_plus_1734.csv
Kauzalni graf: 1000/1734

SP_A | lambda: 0.5 | gravitaciona veza: 0.005 | spoljna veza: 0.1
SP_A | SP istorija: 1000/1734

loto7_4712_k87_loto_plus_1734.csv | SP_A | validacija 1
Istorijskih prelaza: 1039
Pocetak 1 | NLL: 18.243188700 | poziva: 32 | projektovani grad: 7.531e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.251112441 | poziva: 16 | projektovani grad: 1.430e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.256200018 | poziva: 16 | projektovani grad: 3.563e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.629546230

loto7_4712_k87_loto_plus_1734.csv | SP_A | validacija 2
Istorijskih prelaza: 1212
Pocetak 1 | NLL: 18.297802148 | poziva: 11 | projektovani grad: 4.825e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.292814454 | poziva: 15 | projektovani grad: 4.544e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.312434038 | poziva: 18 | projektovani grad: 8.409e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.548787552

SP_B | lambda: 0.9 | gravitaciona veza: 0.01 | spoljna veza: 0.18
SP_B | SP istorija: 1000/1734

loto7_4712_k87_loto_plus_1734.csv | SP_B | validacija 1
Istorijskih prelaza: 1039
Pocetak 1 | NLL: 18.457378119 | poziva: 26 | projektovani grad: 2.341e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.457378119 | poziva: 15 | projektovani grad: 1.718e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.457378119 | poziva: 32 | projektovani grad: 2.254e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.608142009

loto7_4712_k87_loto_plus_1734.csv | SP_B | validacija 2
Istorijskih prelaza: 1212
Pocetak 1 | NLL: 18.478849225 | poziva: 10 | projektovani grad: 2.797e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.478849225 | poziva: 13 | projektovani grad: 4.780e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.478849225 | poziva: 19 | projektovani grad: 5.497e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.559507611

SP_C | lambda: 1.3 | gravitaciona veza: 0.02 | spoljna veza: 0.26
SP_C | SP istorija: 1000/1734

loto7_4712_k87_loto_plus_1734.csv | SP_C | validacija 1
Istorijskih prelaza: 1039
Pocetak 1 | NLL: 18.469261495 | poziva: 17 | projektovani grad: 6.196e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.386209476 | poziva: 22 | projektovani grad: 2.664e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.370733162 | poziva: 41 | projektovani grad: 7.055e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 18.570502410

loto7_4712_k87_loto_plus_1734.csv | SP_C | validacija 2
Istorijskih prelaza: 1212
Pocetak 1 | NLL: 18.396674268 | poziva: 20 | projektovani grad: 2.619e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.408965961 | poziva: 21 | projektovani grad: 3.087e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.402134782 | poziva: 24 | projektovani grad: 1.122e-06
Status: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
Validacioni NLL: 18.444318705

Izabrana dinamika: SP_C
Objedinjeni validacioni NLL: 18.507228736

loto7_4712_k87_loto_plus_1734.csv | trening pre odvojenog testa
Istorijskih prelaza: 1386
Pocetak 1 | NLL: 18.402146762 | poziva: 11 | projektovani grad: 3.472e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.401745755 | poziva: 17 | projektovani grad: 3.590e-08
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 3 | NLL: 18.420697474 | poziva: 18 | projektovani grad: 2.970e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Odvojeni test NLL: 18.361535687
Test je sekvencijalan. Koeficijenti ostaju fiksni, a polja vide samo prethodno ostvarena kola.

loto7_4712_k87_loto_plus_1734.csv | zavrsno ucenje na celoj istoriji
Istorijskih prelaza: 1733
Pocetak 1 | NLL: 18.393289889 | poziva: 14 | projektovani grad: 9.099e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Pocetak 2 | NLL: 18.398444441 | poziva: 16 | projektovani grad: 3.710e-06
Status: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
Pocetak 3 | NLL: 18.414785439 | poziva: 24 | projektovani grad: 3.085e-07
Status: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Pocinjem iscrpni racun zajednickog NEXT maksimuma.
NEXT pretraga: 2,031,616/15,380,937 | proteklo: 0.9 s
NEXT pretraga: 4,063,232/15,380,937 | proteklo: 1.8 s
NEXT pretraga: 6,029,312/15,380,937 | proteklo: 2.6 s
NEXT pretraga: 8,060,928/15,380,937 | proteklo: 3.5 s
NEXT pretraga: 10,027,008/15,380,937 | proteklo: 4.3 s
NEXT pretraga: 12,058,624/15,380,937 | proteklo: 5.2 s
NEXT pretraga: 14,024,704/15,380,937 | proteklo: 6.1 s

loto7_4712_k87_loto_plus_1734.csv
NEXT: 4 x 15 y 19 z 27
Dinamika: SP_C
Validacioni NLL: 18.507228736
Odvojeni test NLL: 18.361535687
Modelna verovatnoca NEXT: 3.71414578168e-05
Zbir svih verovatnoca: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greska Qiskit zajednicke amplitude: 6.776e-21
Efekat samopotencijala u narednom koraku: 1.567431e-04
Promena marginala prema ranijem kontekstu: 7.977290e-02
Osetljivost narednog koraka na finiju integraciju: 1.581150e-06
Koherentni koeficijent 1: 0.567924+0.000000j
Koherentni koeficijent 2: 0.033212+0.573594j
Koherentni koeficijent 3: 0.496802-0.317074j
Presek sa kolom unazad 1: 1/7
Presek sa kolom unazad 2: 0/7
Presek sa kolom unazad 3: 2/7
Dijagnostika ne menja NEXT i ne potvrdjuje prediktivni kvalitet.

ZAVRSNI NEXT REZULTATI
loto7_4712_k87.csv | NEXT: 2 x 14 y 19 z 34 | dinamika: SP_C | test NLL: 18.510931827
Status: optimizator prijavio konvergenciju izabranog kandidata | Max|grad|: 8.883239e-08 | Max|projektovanigrad|: 8.883239e-08
loto7_4712_k87_loto_2978.csv | NEXT: 8 x 19 y 29 z 34 | dinamika: SP_C | test NLL: 18.476708429
Status: optimizator prijavio konvergenciju izabranog kandidata | Max|grad|: 9.230199e-07 | Max|projektovanigrad|: 9.230199e-07
loto7_4712_k87_loto_plus_1734.csv | NEXT: 4 x 15 y 19 z 27 | dinamika: SP_C | test NLL: 18.361535687
Status: optimizator prijavio konvergenciju izabranog kandidata | Max|grad|: 9.098591e-07 | Max|projektovanigrad|: 9.098591e-07
"""



"""
Schrodinger-Poisson model tamne materije. 
Njegova posebnost je povratna sprega: gustina talasnog polja stvara potencijal, a taj potencijal zatim menja evoluciju istog polja.

Dva izvora:
- Cappelli i saradnici, 2024: From Vlasov-Poisson to Schrödinger-Poisson: dark matter simulation with a quantum variational time evolution algorithm. 
Rad definiše kvantna kola i postupak evolucije sa potencijalom koji zavisi od trenutnog stanja. 
- PyUltraLight: stvarni evolve() kod. 
On smenjuje evoluciju pod potencijalom, evoluciju u Furijeovom prostoru i ponovno računanje potencijala iz nove gustine. 
To je klasični numerički solver, koristan za proveru Qiskit adaptacije. 

Za Loto istraživanje te konkretne dinamike:
1. Stanje predstavlja amplitude celih legalnih kombinacija.
2. Trenutna raspodela određuje potencijal preko diskretne Poissonove jednačine.
3. Potencijal i operator širenja zajedno menjaju amplitude narednog stanja.
4. Parametri dinamike uče se iz istorijskih prelaza, hronološki.

Susedstvo se definiše između kombinacija koje se razlikuju zamenom jednog broja. 
Tako ne pretpostavljam da su, na primer, 20 i 21 fizički bliski.

Potencijal se ponovo računa iz razvijajuće raspodele i neposredno upravlja njenom daljom evolucijom. 
To mora ostati suština implementacije. Nelinearnost bi zahtevala jasno označen hibridni postupak sa klasičnim ažuriranjem potencijala.
Posebna matematička ideja. 

Schrödinger-Poisson evolucija: gustina polja određuje potencijal, a taj potencijal menja naredno kvantno stanje. 
Zajednička raspodela sedam brojeva mora da zadrži interferenciju celih kombinacija, bez izbora sedam najvećih pojedinačnih težina.

Prostornu mrežu zamenjuje graf 39 brojeva. 
Veze se računaju samo iz do tada poznatih kola; ne pretpostavljam da su susedni brojevi fizički bliski. 
Poissonova jednačina na tom grafu određuje potencijal za Qiskit evoluciju. 
Proverene su formule za normalizaciju zajedničke raspodele, njen gradijent i očuvanje norme. 

Osnova je Schrödinger-Poisson dinamika sa zajedničkim potencijalom više polja. Kvantni postupak, model više polja.
Adaptacija koristi kauzalni graf 39 brojeva, Qiskit evoluciju polja i koherentnu amplitudu cele kombinacije. 
Parametri dinamike biraju se hronološkom validacijom. 
NEXT se nalazi iscrpnom pretragom svih 15.380.937 kombinacija.
Proverene su referentne matematičke formule. Ugrađene provere prethode treningu.
"""
