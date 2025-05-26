OPENQASM 2.0;
include "qelib1.inc";
gate quantum_volume__1_1_83_ q0 {  }
qreg q[1];
creg meas[1];
quantum_volume__1_1_83_ q[0];
barrier q[0];
barrier q[0];
measure q[0] -> meas[0];
