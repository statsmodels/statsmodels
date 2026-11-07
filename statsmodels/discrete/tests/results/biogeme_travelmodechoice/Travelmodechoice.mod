// Logit model

[ModelDescription]
"Example"

[Choice]
choice

[Beta]
// Name Value  LowerBound UpperBound  status (0=variable, 1=fixed)
ASC_AIR 	0 -10              10              0
ASC_TRAIN  	0 -10              10              0
ASC_BUS	    0 -10              10              0
ASC_CAR     0 -10              10              1
B_TTME		0 -10              10              0
B_GC		0 -10              10              0
B_HINC_AIR  0 -10              10              0

[LaTeX]
ASC_AIR "Intercept(air)"
ASC_TRAIN  "Intercept(train)"
ASC_CAR "Intercept(bus)"
B_TTME  "ttme"
B_GC "gc"
B_HINC_AIR "hinc_air"

[Utilities]
// Id Name     Avail       linear-in-parameter expression (beta1*x1 + beta2*x2 + ... )
	1 V1 		AIR_AV 		ASC_AIR * one +   B_GC * gc_air   + B_TTME * ttme_air + B_HINC_AIR * hinc_air_air
	2 V2 		TRAIN_AV	ASC_TRAIN * one + B_GC * gc_train + B_TTME * ttme_train
	3 V3 		BUS_AV		ASC_BUS  * one  + B_GC * gc_bus   + B_TTME * ttme_bus
	4 V4 		CAR_AV		ASC_CAR  * one  + B_GC * gc_car   + B_TTME * ttme_car

[Expressions]
// Define here arithmetic expressions for name that are not directly
// available from the data
one = 1

[Model]
// $MNL stands for "multinomial logit model"
$MNL
