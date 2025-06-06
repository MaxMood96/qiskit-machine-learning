# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

"""Test Sampler Gradients"""

import unittest
from test import QiskitAlgorithmsTestCase
from typing import List
import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2, real_amplitudes
from qiskit.circuit.library.standard_gates import RXXGate
from qiskit.primitives import Sampler
from qiskit.result import QuasiDistribution
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import Session, SamplerV2

from qiskit_machine_learning.gradients import (
    LinCombSamplerGradient,
    ParamShiftSamplerGradient,
    SPSASamplerGradient,
)

from .logging_primitives import LoggingSampler

gradient_factories = [
    ParamShiftSamplerGradient,
    LinCombSamplerGradient,
]


@ddt
class TestSamplerGradient(QiskitAlgorithmsTestCase):
    """Test Sampler Gradient"""

    def __init__(self, TestCase):
        self.sampler = Sampler()
        super().__init__(TestCase)

    @data(*gradient_factories)
    def test_single_circuit(self, grad):
        """Test the sampler gradient for a single circuit"""

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_p(self, grad):
        """Test the sampler gradient for p"""

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_u(self, grad):
        """Test the sampler gradient for u"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}, {0: 0, 1: 0}, {0: 0, 1: 0}],
            [{0: -0.176777, 1: 0.176777}, {0: -0.426777, 1: 0.426777}, {0: -0.426777, 1: 0.426777}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_efficient_su2(self, grad):
        """Test the sampler gradient for EfficientSU2"""

        qc = efficient_su2(2, reps=1)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        expected = [
            [
                {
                    0: -0.11963834764831836,
                    1: -0.05713834764831845,
                    2: -0.21875000000000003,
                    3: 0.39552669529663675,
                },
                {
                    0: -0.32230339059327373,
                    1: -0.031250000000000014,
                    2: 0.2339150429449554,
                    3: 0.11963834764831843,
                },
                {
                    0: 0.012944173824159189,
                    1: -0.01294417382415923,
                    2: 0.07544417382415919,
                    3: -0.07544417382415919,
                },
                {
                    0: 0.2080266952966367,
                    1: -0.03125000000000002,
                    2: -0.11963834764831842,
                    3: -0.057138347648318405,
                },
                {
                    0: -0.11963834764831838,
                    1: 0.11963834764831838,
                    2: -0.21875000000000003,
                    3: 0.21875,
                },
                {
                    0: -0.2781092167691146,
                    1: -0.0754441738241592,
                    2: 0.27810921676911443,
                    3: 0.07544417382415924,
                },
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
            [
                {
                    0: -4.163336342344337e-17,
                    1: 2.7755575615628914e-17,
                    2: -4.163336342344337e-17,
                    3: 0.0,
                },
                {0: 0.0, 1: -1.3877787807814457e-17, 2: 4.163336342344337e-17, 3: 0.0},
                {
                    0: -0.24999999999999994,
                    1: 0.24999999999999994,
                    2: 0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {
                    0: 0.24999999999999994,
                    1: 0.24999999999999994,
                    2: -0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {
                    0: -4.163336342344337e-17,
                    1: 4.163336342344337e-17,
                    2: -4.163336342344337e-17,
                    3: 5.551115123125783e-17,
                },
                {
                    0: -0.24999999999999994,
                    1: 0.24999999999999994,
                    2: 0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {0: 0.0, 1: 2.7755575615628914e-17, 2: 0.0, 3: 2.7755575615628914e-17},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(expected[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_2qubit_gate(self, grad):
        """Test the sampler gradient for 2 qubit gates"""

        for gate in [RXXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0, 3: 0.5 / np.sqrt(2)}],
                [{0: -0.5, 1: 0, 2: 0, 3: 0.5}],
            ]
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                qc.measure_all()
                gradient = grad(self.sampler)
                gradients = gradient.run([qc], [param]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=2)
                array2 = _quasi2array(correct_results[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameter_coefficient(self, grad):
        """Test the sampler gradient for parameter variables with coefficients"""

        qc = real_amplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [
                {
                    0: 0.30014831912265927,
                    1: -0.6634809704357856,
                    2: 0.343589357193753,
                    3: 0.019743294119373426,
                },
                {
                    0: 0.16470607453981906,
                    1: -0.40996282450610577,
                    2: 0.08791803062881773,
                    3: 0.15733871933746948,
                },
                {
                    0: 0.27036068339663866,
                    1: -0.273790986018701,
                    2: 0.12752010079553433,
                    3: -0.12408979817347202,
                },
                {
                    0: -0.2098616294167757,
                    1: -0.2515823946449894,
                    2: 0.21929102305386305,
                    3: 0.24215300100790207,
                },
            ],
            [
                {
                    0: -1.844810060881004,
                    1: 0.04620532700836027,
                    2: 1.6367366426074323,
                    3: 0.16186809126521057,
                },
                {
                    0: 0.07296073407769421,
                    1: -0.021774869186331716,
                    2: 0.02177486918633173,
                    3: -0.07296073407769456,
                },
                {
                    0: -0.07794369186049102,
                    1: -0.07794369186049122,
                    2: 0.07794369186049117,
                    3: 0.07794369186049112,
                },
                {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
            ],
        ]

        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(correct_results[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameters(self, grad):
        """Test the sampler gradient for parameters"""

        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rz(b, 0)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4, np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param], parameters=[[a]]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            qc.measure_all()
            param_values = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            params = [[a, b, c], [c, b, a], [a, c], [c, a]]
            expected = [
                [
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                    {0: 0.3061861668168149, 1: -0.3061861668167012},
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                ],
                [
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                    {0: 0.3061861668168149, 1: -0.3061861668167012},
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                ],
                [
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                ],
                [
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                ],
            ]
            for i, p in enumerate(params):  # pylint: disable=invalid-name
                gradients = gradient.run([qc], param_values, parameters=[p]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=1)
                array2 = _quasi2array(expected[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_multi_arguments(self, grad):
        """Test the sampler gradient for multiple arguments"""

        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        qc2.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        gradients = gradient.run([qc, qc2], param_list).result().gradients
        for i, q_dists in enumerate(gradients):
            array1 = _quasi2array(q_dists, num_qubits=1)
            array2 = _quasi2array(correct_results[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

        # parameters
        with self.subTest(msg="Different parameters"):
            c = Parameter("c")
            qc3 = QuantumCircuit(1)
            qc3.rx(c, 0)
            qc3.ry(a, 0)
            qc3.measure_all()
            param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
            gradients = (
                gradient.run([qc, qc3, qc3], param_list2, parameters=[[a], [c], None])
                .result()
                .gradients
            )
            correct_results = [
                [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
                [{0: -0.25, 1: 0.25}],
                [{0: -0.25, 1: 0.25}, {0: -0.25, 1: 0.25}],
            ]
            for i, q_dists in enumerate(gradients):
                array1 = _quasi2array(q_dists, num_qubits=1)
                array2 = _quasi2array(correct_results[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_validation(self, grad):
        """Test sampler gradient's validation"""

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        gradient = grad(self.sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        with self.assertRaises(ValueError):
            gradient.run([qc], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [[np.pi / 4, np.pi / 4]])

    def test_spsa_gradient(self):
        """Test the SPSA sampler gradient"""

        with self.assertRaises(ValueError):
            _ = SPSASamplerGradient(self.sampler, epsilon=-0.1)

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        qc.measure_all()
        param_list = [[1, 2]]
        correct_results = [
            [
                {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
            ],
        ]
        gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, seed=123)
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(correct_results[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

        # multi parameters
        with self.subTest(msg="Multiple parameters"):
            param_list2 = [[1, 2], [1, 2], [3, 4]]
            correct_results2 = [
                [
                    {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                    {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
                ],
                [
                    {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
                ],
                [
                    {0: -0.0141129, 1: -0.0564471, 2: -0.3642884, 3: 0.4348484},
                    {0: 0.0141129, 1: 0.0564471, 2: 0.3642884, 3: -0.4348484},
                ],
            ]
            gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, seed=123)
            gradients = (
                gradient.run([qc] * 3, param_list2, parameters=[None, [b], None]).result().gradients
            )
            for i, result in enumerate(gradients):
                array1 = _quasi2array(result, num_qubits=2)
                array2 = _quasi2array(correct_results2[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

        # batch size
        with self.subTest(msg="Batch size"):
            param_list = [[1, 1]]
            gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, batch_size=4, seed=123)
            gradients = gradient.run([qc], param_list).result().gradients
            correct_results3 = [
                [
                    {
                        0: -0.1620149622932887,
                        1: -0.25872053011771756,
                        2: 0.3723827084675668,
                        3: 0.04835278392088804,
                    },
                    {
                        0: -0.1620149622932887,
                        1: 0.3723827084675668,
                        2: -0.25872053011771756,
                        3: 0.04835278392088804,
                    },
                ]
            ]
            for i, q_dists in enumerate(gradients):
                array1 = _quasi2array(q_dists, num_qubits=2)
                array2 = _quasi2array(correct_results3[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            qc.measure_all()
            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            correct_results = [
                [
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                ],
                [
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                ],
                [
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                ],
                [
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                ],
            ]
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, seed=123)
                gradients = gradient.run([qc], param_list, parameters=[p]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=1)
                array2 = _quasi2array(correct_results[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(
        ParamShiftSamplerGradient,
        LinCombSamplerGradient,
        SPSASamplerGradient,
    )
    def test_operations_preserved(self, gradient_cls):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.initialize(np.array([1, 1, 0, 0]) / np.sqrt(2))  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled
        circuit.measure_all()

        values = [np.pi / 2]
        expect = [{0: 0, 1: -0.25, 2: 0, 3: 0.25}]

        ops = []

        def operations_callback(op):
            ops.append(op)

        sampler = LoggingSampler(operations_callback=operations_callback)

        if gradient_cls in [SPSASamplerGradient]:
            gradient = gradient_cls(sampler, epsilon=0.01)
        else:
            gradient = gradient_cls(sampler)

        job = gradient.run([circuit], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            array1 = _quasi2array(result.gradients[0], num_qubits=2)
            array2 = _quasi2array(expect, num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-5)


@ddt
class TestSamplerGradientV2(QiskitAlgorithmsTestCase):
    """Test Sampler Gradient"""

    def __init__(self, TestCase):
        backend = GenericBackendV2(num_qubits=3, seed=123)
        session = Session(backend=backend)
        self.sampler = SamplerV2(mode=session)
        self.pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        super().__init__(TestCase)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_single_circuit(self, grad):
        """Test the sampler gradient for a single circuit"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()

        gradient = grad(sampler=self.sampler, pass_manager=self.pass_manager)
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=0.5, rtol=0.5)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_p(self, grad):
        """Test the sampler gradient for p"""

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()

        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_u(self, grad):
        """Test the sampler gradient for u"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}, {0: 0, 1: 0}, {0: 0, 1: 0}],
            [{0: -0.176777, 1: 0.176777}, {0: -0.426777, 1: 0.426777}, {0: -0.426777, 1: 0.426777}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_efficient_su2(self, grad):
        """Test the sampler gradient for EfficientSU2"""

        qc = efficient_su2(2, reps=1)
        qc.measure_all()
        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        expected = [
            [
                {
                    0: -0.11963834764831836,
                    1: -0.05713834764831845,
                    2: -0.21875000000000003,
                    3: 0.39552669529663675,
                },
                {
                    0: -0.32230339059327373,
                    1: -0.031250000000000014,
                    2: 0.2339150429449554,
                    3: 0.11963834764831843,
                },
                {
                    0: 0.012944173824159189,
                    1: -0.01294417382415923,
                    2: 0.07544417382415919,
                    3: -0.07544417382415919,
                },
                {
                    0: 0.2080266952966367,
                    1: -0.03125000000000002,
                    2: -0.11963834764831842,
                    3: -0.057138347648318405,
                },
                {
                    0: -0.11963834764831838,
                    1: 0.11963834764831838,
                    2: -0.21875000000000003,
                    3: 0.21875,
                },
                {
                    0: -0.2781092167691146,
                    1: -0.0754441738241592,
                    2: 0.27810921676911443,
                    3: 0.07544417382415924,
                },
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
            [
                {
                    0: -4.163336342344337e-17,
                    1: 2.7755575615628914e-17,
                    2: -4.163336342344337e-17,
                    3: 0.0,
                },
                {0: 0.0, 1: -1.3877787807814457e-17, 2: 4.163336342344337e-17, 3: 0.0},
                {
                    0: -0.24999999999999994,
                    1: 0.24999999999999994,
                    2: 0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {
                    0: 0.24999999999999994,
                    1: 0.24999999999999994,
                    2: -0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {
                    0: -4.163336342344337e-17,
                    1: 4.163336342344337e-17,
                    2: -4.163336342344337e-17,
                    3: 5.551115123125783e-17,
                },
                {
                    0: -0.24999999999999994,
                    1: 0.24999999999999994,
                    2: 0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {0: 0.0, 1: 2.7755575615628914e-17, 2: 0.0, 3: 2.7755575615628914e-17},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(expected[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=0.5, rtol=0.5)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_2qubit_gate(self, grad):
        """Test the sampler gradient for 2 qubit gates"""

        for gate in [RXXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0, 3: 0.5 / np.sqrt(2)}],
                [{0: -0.5, 1: 0, 2: 0, 3: 0.5}],
            ]
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                qc.measure_all()
                gradient = grad(sampler=self.sampler, pass_manager=self.pass_manager)
                gradients = gradient.run([qc], [param]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=2)
                array2 = _quasi2array(correct_results[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_parameter_coefficient(self, grad):
        """Test the sampler gradient for parameter variables with coefficients"""

        qc = real_amplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        qc.measure_all()

        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [
                {
                    0: 0.30014831912265927,
                    1: -0.6634809704357856,
                    2: 0.343589357193753,
                    3: 0.019743294119373426,
                },
                {
                    0: 0.16470607453981906,
                    1: -0.40996282450610577,
                    2: 0.08791803062881773,
                    3: 0.15733871933746948,
                },
                {
                    0: 0.27036068339663866,
                    1: -0.273790986018701,
                    2: 0.12752010079553433,
                    3: -0.12408979817347202,
                },
                {
                    0: -0.2098616294167757,
                    1: -0.2515823946449894,
                    2: 0.21929102305386305,
                    3: 0.24215300100790207,
                },
            ],
            [
                {
                    0: -1.844810060881004,
                    1: 0.04620532700836027,
                    2: 1.6367366426074323,
                    3: 0.16186809126521057,
                },
                {
                    0: 0.07296073407769421,
                    1: -0.021774869186331716,
                    2: 0.02177486918633173,
                    3: -0.07296073407769456,
                },
                {
                    0: -0.07794369186049102,
                    1: -0.07794369186049122,
                    2: 0.07794369186049117,
                    3: 0.07794369186049112,
                },
                {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
            ],
        ]

        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(correct_results[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=0.5, rtol=0.5)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_parameters(self, grad):
        """Test the sampler gradient for parameters"""

        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rz(b, 0)
        qc.measure_all()

        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [[np.pi / 4, np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param], parameters=[[a]]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=0.5, rtol=0.5)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            qc.measure_all()

            param_values = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            params = [[a, b, c], [c, b, a], [a, c], [c, a]]
            expected = [
                [
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                    {0: 0.3061861668168149, 1: -0.3061861668167012},
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                ],
                [
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                    {0: 0.3061861668168149, 1: -0.3061861668167012},
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                ],
                [
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                ],
                [
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                ],
            ]
            for i, p in enumerate(params):  # pylint: disable=invalid-name
                gradients = gradient.run([qc], param_values, parameters=[p]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=1)
                array2 = _quasi2array(expected[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    @unittest.skip("Skipping due to noise sensitivity.")
    def test_gradient_multi_arguments(self, grad):
        """Test the sampler gradient for multiple arguments"""

        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()

        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        qc2.measure_all()
        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        gradients = gradient.run([qc, qc2], param_list).result().gradients
        for i, q_dists in enumerate(gradients):
            array1 = _quasi2array(q_dists, num_qubits=1)
            array2 = _quasi2array(correct_results[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

        # parameters
        with self.subTest(msg="Different parameters"):
            c = Parameter("c")
            qc3 = QuantumCircuit(1)
            qc3.rx(c, 0)
            qc3.ry(a, 0)
            qc3.measure_all()
            param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
            gradients = (
                gradient.run([qc, qc3, qc3], param_list2, parameters=[[a], [c], None])
                .result()
                .gradients
            )
            correct_results = [
                [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
                [{0: -0.25, 1: 0.25}],
                [{0: -0.25, 1: 0.25}, {0: -0.25, 1: 0.25}],
            ]
            for i, q_dists in enumerate(gradients):
                array1 = _quasi2array(q_dists, num_qubits=1)
                array2 = _quasi2array(correct_results[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    def test_gradient_validation(self, grad):
        """Test sampler gradient's validation"""

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()

        gradient = grad(
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        param_list = [[np.pi / 4], [np.pi / 2]]
        with self.assertRaises(ValueError):
            gradient.run([qc], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [[np.pi / 4, np.pi / 4]])

    @unittest.skip("Skipping due to noise sensitivity.")
    def test_spsa_gradient(self):
        """Test the SPSA sampler gradient"""

        with self.assertRaises(ValueError):
            _ = SPSASamplerGradient(self.sampler, epsilon=-0.01)

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        qc.measure_all()
        param_list = [[1, 2]]
        correct_results = [
            [
                {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
            ],
        ]
        gradient = SPSASamplerGradient(
            sampler=self.sampler, pass_manager=self.pass_manager, epsilon=1e-6, seed=123
        )
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(correct_results[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-3)

        # multi parameters
        with self.subTest(msg="Multiple parameters"):
            param_list2 = [[1, 2], [1, 2], [3, 4]]
            correct_results2 = [
                [
                    {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                    {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
                ],
                [
                    {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
                ],
                [
                    {0: -0.0141129, 1: -0.0564471, 2: -0.3642884, 3: 0.4348484},
                    {0: 0.0141129, 1: 0.0564471, 2: 0.3642884, 3: -0.4348484},
                ],
            ]
            gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, seed=123)
            gradients = (
                gradient.run([qc] * 3, param_list2, parameters=[None, [b], None]).result().gradients
            )
            for i, result in enumerate(gradients):
                array1 = _quasi2array(result, num_qubits=2)
                array2 = _quasi2array(correct_results2[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

        # batch size
        with self.subTest(msg="Batch size"):
            param_list = [[1, 1]]
            gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, batch_size=4, seed=123)
            gradients = gradient.run([qc], param_list).result().gradients
            correct_results3 = [
                [
                    {
                        0: -0.1620149622932887,
                        1: -0.25872053011771756,
                        2: 0.3723827084675668,
                        3: 0.04835278392088804,
                    },
                    {
                        0: -0.1620149622932887,
                        1: 0.3723827084675668,
                        2: -0.25872053011771756,
                        3: 0.04835278392088804,
                    },
                ]
            ]
            for i, q_dists in enumerate(gradients):
                array1 = _quasi2array(q_dists, num_qubits=2)
                array2 = _quasi2array(correct_results3[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            qc.measure_all()
            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            correct_results = [
                [
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                ],
                [
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                ],
                [
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                ],
                [
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                ],
            ]
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = SPSASamplerGradient(self.sampler, epsilon=1e-6, seed=123)
                gradients = gradient.run([qc], param_list, parameters=[p]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=1)
                array2 = _quasi2array(correct_results[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-3)

    @data(
        ParamShiftSamplerGradient,
        LinCombSamplerGradient,
        SPSASamplerGradient,
    )
    def test_operations_preserved(self, gradient_cls):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.initialize(np.array([1, 1, 0, 0]) / np.sqrt(2))  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled
        circuit.measure_all()

        values = [np.pi / 2]
        expect = [{0: 0, 1: -0.25, 2: 0, 3: 0.25}]

        ops = []

        def operations_callback(op):
            ops.append(op)

        sampler = LoggingSampler(operations_callback=operations_callback)

        if gradient_cls in [SPSASamplerGradient]:
            gradient = gradient_cls(sampler, epsilon=0.01)
        else:
            gradient = gradient_cls(sampler)

        job = gradient.run([circuit], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            array1 = _quasi2array(result.gradients[0], num_qubits=2)
            array2 = _quasi2array(expect, num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-5)


def _quasi2array(quasis: List[QuasiDistribution], num_qubits: int) -> np.ndarray:
    ret = np.zeros((len(quasis), 2**num_qubits))
    for i, quasi in enumerate(quasis):
        ret[i, list(quasi.keys())] = list(quasi.values())
    return ret


if __name__ == "__main__":
    unittest.main()
