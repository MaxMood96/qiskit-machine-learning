# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator quantum neural network class"""

from __future__ import annotations

import logging
from copy import copy
from typing import Sequence
import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives import BaseEstimator, BaseEstimatorV1, Estimator, EstimatorResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler.passmanager import BasePassManager


from ..gradients import (
    BaseEstimatorGradient,
    EstimatorGradientResult,
    ParamShiftEstimatorGradient,
)

from ..circuit.library import QNNCircuit
from ..exceptions import QiskitMachineLearningError
from ..utils.deprecation import issue_deprecation_msg
from .neural_network import NeuralNetwork

logger = logging.getLogger(__name__)


class EstimatorQNN(NeuralNetwork):
    """A neural network implementation based on the Estimator primitive.

    The ``EstimatorQNN`` is a neural network that takes in a parametrized quantum circuit
    with designated parameters for input data and/or weights, an optional observable(s) and outputs
    their expectation value(s). Quite often, a combined quantum circuit is used. Such a circuit is
    built from two circuits: a feature map, it provides input parameters for the network, and an
    ansatz (weight parameters).
    In this case a :meth:`~qiskit_machine_learning.circuit.library.qnn_circuit` can be used to
    simplify the composition of a feature map and ansatz.
    Use of the :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` being passed as circuit,
    the input and weight parameters do not have to be provided, because these two properties are taken
    from the :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is deprecated.

    Example:

    .. code-block::

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import zz_feature_map, real_amplitudes
        from qiskit_machine_learning.circuit.library import qnn_circuit

        from qiskit_machine_learning.neural_networks import EstimatorQNN

        num_qubits = 2

        # Using the QNNCircuit:
        # Create a parametrized 2 qubit circuit composed of the default zz_feature_map feature map
        # and real_amplitudes ansatz.
        qnn_qc, fm_params, anz_params = qnn_circuit(num_qubits)

        qnn = EstimatorQNN(
            circuit=qnn_qc,
            input_params=fm_params,
            weight_params=anz_params
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])

        # Explicitly specifying the ansatz and feature map:
        feature_map = zz_feature_map(feature_dimension=num_qubits)
        ansatz = real_amplitudes(num_qubits=num_qubits)

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])


    The following attributes can be set via the constructor but can also be read and
    updated once the EstimatorQNN object has been constructed.

    Attributes:

        estimator (BaseEstimator): The estimator primitive used to compute the neural network's results.
        gradient (BaseEstimatorGradient): The estimator gradient to be used for the backward
            pass.
    """

    def __init__(
        self,
        *,
        circuit: QuantumCircuit,
        estimator: BaseEstimator | BaseEstimatorV2 | None = None,
        observables: Sequence[BaseOperator] | BaseOperator | None = None,
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
        gradient: BaseEstimatorGradient | None = None,
        input_gradients: bool = False,
        default_precision: float = 0.015625,
        pass_manager: BasePassManager | None = None,
    ):
        r"""
        Args:
            circuit: The quantum circuit to represent the neural network. If a
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is passed, the
                ``input_params`` and ``weight_params`` do not have to be provided, because these two
                properties are taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` (DEPRECATED).
            estimator: The estimator used to compute neural network's results.
                If ``None``, a default instance of the reference estimator,
                :class:`~qiskit.primitives.Estimator`, will be used.

                .. warning::

                    The assignment ``estimator=None`` defaults to using
                    :class:`~qiskit.primitives.Estimator`, which points to a deprecated estimator V1
                    (as of Qiskit 1.2). ``EstimatorQNN`` will adopt Estimator V2 as default no later than
                    Qiskit Machine Learning 0.9.

            observables: The observables for outputs of the neural network. If ``None``,
                use the default :math:`Z^{\otimes n}` observable, where :math:`n`
                is the number of qubits.
            input_params: The parameters that correspond to the input data of the network.
                If ``None``, the input data is not bound to any parameters.
                If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `input_params` value here is ignored. Instead, the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` input_parameters.
            weight_params: The parameters that correspond to the trainable weights.
                If ``None``, the weights are not bound to any parameters.
                If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `weight_params` value here is ignored. Instead, the value is taken from the
                `weight_parameters` associated with
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`.
            gradient: The estimator gradient to be used for the backward pass.
                If ``None``, a default instance of the estimator gradient,
                :class:`~qiskit_machine_learning.gradients.ParamShiftEstimatorGradient`, will be used.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using
                :class:`~qiskit_machine_learning.connectors.TorchConnector`.
            default_precision: The default precision for the estimator if not specified during run.
            pass_manager: The pass manager to transpile the circuits, if necessary.
                Defaults to ``None``, as some primitives do not need transpiled circuits.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        if estimator is None:
            estimator = Estimator()

        if isinstance(estimator, BaseEstimatorV1):
            issue_deprecation_msg(
                msg="V1 Primitives are deprecated",
                version="0.8.0",
                remedy="Use V2 primitives for continued compatibility and support.",
                period="4 months",
            )
        self.estimator = estimator

        if hasattr(circuit.layout, "_input_qubit_count"):
            self.num_virtual_qubits = circuit.layout._input_qubit_count
        else:
            if pass_manager is None:
                self.num_virtual_qubits = circuit.num_qubits
            else:
                circuit = pass_manager.run(circuit)
                if hasattr(circuit.layout, "_input_qubit_count"):
                    self.num_virtual_qubits = circuit.layout._input_qubit_count
                else:
                    self.num_virtual_qubits = circuit.num_qubits

        self._org_circuit = circuit

        if observables is None:
            observables = SparsePauliOp.from_sparse_list(
                [("Z" * self.num_virtual_qubits, range(self.num_virtual_qubits), 1)],
                num_qubits=self.circuit.num_qubits,
            )

        if isinstance(observables, BaseOperator):
            observables = (observables,)

        self._observables = observables

        if isinstance(circuit, QNNCircuit):
            issue_deprecation_msg(
                msg="Using QNNCircuit here is deprecated",
                version="0.9.0",
                remedy="Use qnn_circuit (instead) of QNNCircuit and pass "
                "explicitly the input and weight parameters.",
                period="4 months",
            )
            self._input_params = list(circuit.input_parameters)
            self._weight_params = list(circuit.weight_parameters)
        else:
            self._input_params = list(input_params) if input_params is not None else []
            self._weight_params = list(weight_params) if weight_params is not None else []

        # set gradient
        if gradient is None:
            if isinstance(estimator, BaseEstimatorV1):
                gradient = ParamShiftEstimatorGradient(estimator=self.estimator)
            else:
                if pass_manager is None:
                    logger.warning(
                        "No gradient function provided, creating a gradient function."
                        " If your Estimator requires transpilation, please provide a pass manager."
                    )
                gradient = ParamShiftEstimatorGradient(
                    estimator=self.estimator, pass_manager=pass_manager
                )
        self._default_precision = default_precision
        self.gradient = gradient
        self._input_gradients = input_gradients

        super().__init__(
            num_inputs=len(self._input_params),
            num_weights=len(self._weight_params),
            sparse=False,
            output_shape=len(self._observables),
            input_gradients=input_gradients,
        )

        self._circuit = self._reparameterize_circuit(circuit, input_params, weight_params)

    @property
    def circuit(self) -> QuantumCircuit:
        """The quantum circuit representing the neural network."""
        return copy(self._org_circuit)

    @property
    def observables(self) -> Sequence[BaseOperator] | BaseOperator:
        """Returns the underlying observables of this QNN."""
        return copy(self._observables)

    @property
    def input_params(self) -> Sequence[Parameter] | None:
        """The parameters that correspond to the input data of the network."""
        return copy(self._input_params)

    @property
    def weight_params(self) -> Sequence[Parameter] | None:
        """The parameters that correspond to the trainable weights."""
        return copy(self._weight_params)

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default, such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off computation of gradients with respect to input data."""
        self._input_gradients = input_gradients

    @property
    def default_precision(self) -> float:
        """Return the default precision"""
        return self._default_precision

    def _forward_postprocess(self, num_samples: int, result: EstimatorResult) -> np.ndarray:
        """Post-processing during forward pass of the network."""
        return np.reshape(result, (-1, num_samples)).T

    def _forward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> np.ndarray | None:
        """Forward pass of the neural network."""
        parameter_values_, num_samples = self._preprocess_forward(input_data, weights)

        # Determine how to run the estimator based on its version
        if isinstance(self.estimator, BaseEstimatorV1):
            job = self.estimator.run(
                [self._circuit] * num_samples * self.output_shape[0],
                [op for op in self._observables for _ in range(num_samples)],
                np.tile(parameter_values_, (self.output_shape[0], 1)),
            )
            results = job.result().values

        elif isinstance(self.estimator, BaseEstimatorV2):

            # Prepare circuit-observable-parameter tuples (PUBs)
            circuit_observable_params = []
            for observable in self._observables:
                circuit_observable_params.append((self._circuit, observable, parameter_values_))

            # For BaseEstimatorV2, run the estimator using PUBs and specified precision
            job = self.estimator.run(circuit_observable_params, precision=self._default_precision)
            results = [result.data.evs for result in job.result()]
        else:
            raise QiskitMachineLearningError(
                "The accepted estimators are BaseEstimatorV1 and BaseEstimatorV2; got "
                + f"{type(self.estimator)} instead. Note that BaseEstimatorV1 is deprecated in"
                + "Qiskit and removed in Qiskit IBM Runtime."
            )
        return self._forward_postprocess(num_samples, results)

    def _backward_postprocess(
        self, num_samples: int, result: EstimatorGradientResult
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Post-processing during backward pass of the network."""
        num_observables = self.output_shape[0]
        if self._input_gradients:
            input_grad = np.zeros((num_samples, num_observables, self._num_inputs))
        else:
            input_grad = None

        weights_grad = np.zeros((num_samples, num_observables, self._num_weights))
        gradients = np.asarray(result.gradients)
        for i in range(num_observables):
            if self._input_gradients:
                input_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples][
                    :, : self._num_inputs
                ]
                weights_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples][
                    :, self._num_inputs :
                ]
            else:
                weights_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples]
        return input_grad, weights_grad

    def _backward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        input_grad, weights_grad = None, None

        if np.prod(parameter_values.shape) > 0:
            num_observables = self.output_shape[0]
            num_circuits = num_samples * num_observables

            circuits = [self._circuit] * num_circuits
            observables = [op for op in self._observables for _ in range(num_samples)]
            param_values = np.tile(parameter_values, (num_observables, 1))

            job = None

            if self._input_gradients:

                job = self.gradient.run(circuits, observables, param_values)

            elif len(parameter_values[0]) > self._num_inputs:
                params = [self._circuit.parameters[self._num_inputs :]] * num_circuits
                job = self.gradient.run(circuits, observables, param_values, parameters=params)

            if job is not None:
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError(f"Estimator job failed. {exc}") from exc

                input_grad, weights_grad = self._backward_postprocess(num_samples, results)

        return input_grad, weights_grad
