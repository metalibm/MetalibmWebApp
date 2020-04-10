
execute("polynomials.sol");

execute("doubleexpansion.sol");

execute("horner.sol");

//f = sin(x);
//f = exp(x);
f = cos(x);
I = [-2^-4;2^-4];
//monomials = [| 1, 3, 5, 7, 9, 11 |];
//monomials = [| 0, ..., 8 |];
monomials = [| 0, 2, 4, 6, 8, 10 |];
prec = 3000!;
p = fpminimax(f, monomials, [|DD,DD,D...|], [1b-100; 2^-4]);
prec = default!;
epsTarget = dirtyinfnorm(p/f-1, I);

gappaFile = "test.gappa";

PR = polynomialToTree(p);

context = { /* The .variableFormats field is an opaque type describing the format of _x_ */
	    .variableFormat = { .bits = 102, .p = _x_, .I = I },
	    
	    /* The methods below are required, EVEN FOR absolute error computations. They may
	       return infinite relative error bounds though.
	    */
	    .compareFormats = compareFormats,
	    .roundConstant = roundConstantToTargetPrecFormatSpecific,
		    
            .computeBoundAddition = computeActualAdditionBound,
            .computeBoundMultiplication = computeActualMultiplicationBound,
	    .computeBoundPower = computeBoundPower,
	    .computeBoundVariableRounding = computeErrorDueToVariableRounding,
	    .computeOutputFormatAddition = computeOutputFormatAddition,	
	    .computeOutputFormatMultiplication = computeOutputFormatMultiplication,
	    .computeOutputFormatPower = computeOutputFormatPower,
	    .computeConstantFormat = computeFormatForConstant,
	    .computeVariableFormat = computeNeededVariableFormat,
	    
	    /* The methods below are optional and used only for absolute error */
	    .computeBoundAdditionAbsolute = computeActualAdditionBoundAbsolute,
	    .computeBoundMultiplicationAbsolute = computeActualMultiplicationBoundAbsolute,
	    .computeBoundPowerAbsolute = computeBoundPowerAbsolute,
	    .computeBoundVariableRoundingAbsolute = computeErrorDueToVariableRoundingAbsolute,
	    .computeOutputFormatAdditionAbsolute = computeOutputFormatAdditionAbsolute,
	    .computeOutputFormatMultiplicationAbsolute = computeOutputFormatMultiplicationAbsolute,
	    .computeOutputFormatPowerAbsolute = computeOutputFormatPowerAbsolute,
	    .computeVariableFormatAbsolute = computeNeededVariableFormatAbsolute
	  };
	    
if (PR.okay) then {
   P = PR.poly;
   R = computeErrorBounds(P, I, epsTarget, absolute, context);
   print("result = ", R);
   /*
   if (R.okay) then {
      Gappa = generateGappa("foo_", R.poly, R.powerings, context);
      print("Gappa = ", Gappa);
      if (Gappa.okay) then {
      	 write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n", Gappa.gappa, "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
	 "epsTarget = 2^(", log2(abs(epsTarget)), ")";
	 "epsActual = 2^(", log2(abs(R.poly.eps)), ")";
	 write(Gappa.gappa, "\n") > gappaFile;
      };
   };
   */
};

