#!/bin/bash

../build/ode_treefile --DataT="real_data/data_t.txt" --DataX="real_data/data_x.txt" --TreeFile="example4.json" --ConstantUpperBound=30 --ConstantLowerBound=0 --MaxEpoch=5000 --WhenPenalty=1 --PreTraining=200 --OperatorSetting="op.ini" --Epsilon=0.2
