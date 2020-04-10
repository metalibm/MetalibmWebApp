
execute("polynomials.sol");

execute("ml_expansion.sol");

execute("horner.sol");

f = exp(x);
I = [-2^-4;2^-4];
n = 9;
// p = fpminimax(f, n, [|DD,DD,D...|], I);
// epsTarget = dirtyinfnorm(p/f-1, I);

p = 0.125 * x + 3;
epsTarget=2^-70;

PR = polynomialToTree(p);


// description of Metalibm's Lugdunum format object
mll_format = {
    .ml_precision = "dd", // ML_DoubleDouble
    .accuracy = 102, .// accuracy (upper bound on the relative error, in bits)
    overlap=[0] // max overlap [n <- n+1]
};

mll_context = { .variableFormat = mll_format,
	    .compareFormats = mll_compareFormats,
        .computeBoundAddition = mll_computeActualAdditionBound,
        .computeBoundMultiplication = mll_computeActualMultiplicationBound,
	    .computeBoundVariableRounding = mll_computeErrorDueToVariableRounding,
	    .computeOutputFormatAddition = mll_computeOutputFormatAddition,	
	    .computeOutputFormatMultiplication = mll_computeOutputFormatMultiplication,
	    .computeConstantFormat = mll_computeFormatForConstant,
	    .computeVariableFormat = mll_computeNeededVariableFormat,
	    .roundConstant = mll_roundConstantToTargetPrecFormatSpecific	
};

if (PR.okay) then {
   P = PR.poly;
   R = computeErrorBounds(P, I, epsTarget, mll_context);
   print("result = ", R);
};

