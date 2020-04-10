/* Return a bound on the relative error for a multiplication
   multiplying an inputFormatA variable by an inputFormatB variable,
   returning an outputFormatVariable.

   The input types are opaque format types.
   The output type is a positive or zero real number
*/
procedure mll_computeActualMultiplicationBound(outputFormat, inputFormatA, inputFormatB) {
    // the accuracy is the max of the product of inputs accuracies
    // and the output format accuracy
	bound = max(2^(-outputFormat.accuracy), 2^(-inputFormatA.accuracy) * 2^(-inputFormatB.accuracy));
    "mult bound";
    bound;
    return bound;
};


/** return an opaque format object which can fit a result
 *  whose relative error is accuracy (in bits)
 */
procedure mll_getFormatFromAccuracy(accuracy) {
    var res;
    "mll_getFormatFromAccuracy";
    accuracy;
    if (accuracy <= 24) then {
        res =  {.ml_precision = "single", .accuracy = accuracy};
    } else {
        if (accuracy <= 53) then {
            res = {.ml_precision = "double", .accuracy = accuracy};
        } else {
            // check accuracy bound from double - dd selection
            if (accuracy <= 105) then {
                res = {.ml_precision = "dd", .accuracy = accuracy};
            } else {
                if (accuracy <= 150) then { res = {.ml_precision = "td", .accuracy = accuracy};}
                else { 
                    res = error; 
				};
            };
        };
    };

    return res;
};

17.0;

/* Return a bound on the relative error for an addition
   adding an inputFormatA variable by an inputFormatB variable,
   returning an outputFormatVariable.

   The input types are opaque format types.
   The output type is a positive or zero real number
*/
procedure mll_computeActualAdditionBound(outputFormat, inputFormatA, inputFormatB) {
	  bound = 2^(-max((-outputFormat.accuracy), inputFormatA.accuracy, inputFormatB.accuracy));
      "add bound";
      bound;
      return bound;
};

/* Returns the output format of a multiplication that will be fed two operands
   of formats inputFormatA and inputFormatB and that is supposed to have a
   relative error less than or equal to epsTarget.

   The input and output formats are of type "opaque format type".
   The epsTarget input is a positive or zero real number.

*/
procedure mll_computeOutputFormatMultiplication(epsTarget, inputFormatA, inputFormatB) {
	  var res;

	  if (epsTarget > 0) then {
	     if ((epsTarget >= 2^(-53)) && (inputFormatA.accuracy <= 53) && (inputFormatB.accuracy <= 53)) then {
               res = {.ml_precision = "double", .accuracy = 53, .overlap = [0]};
	     } else {
	       accuracy = max(inputFormatA.accuracy, inputFormatB.accuracy, ceil(max(2, ceil(-log2(epsTarget)) / 51) * 51));
           "mll_computeOutputFormatMultiplication 1";
           accuracy;
           res = mll_getFormatFromAccuracy(accuracy);
	     };
	  } else {
	     accuracy = max(inputFormatA.accuracy, inputFormatB.accuracy);
          "mll_computeOutputFormatMultiplication 2";
           accuracy;
         res = mll_getFormatFromAccuracy(accuracy);
	  };
	
	  return res;
};

/* Returns the output format of an addition  that will be fed two operands
   of formats inputFormatA and inputFormatB and that is supposed to have a
   relative error less than or equal to epsTarget.

   The input and output formats are of type "opaque format type".
   The epsTarget input is a positive or zero real number.

*/
procedure mll_computeOutputFormatAddition(epsTarget, inputFormatA, inputFormatB) {
	  var res;

	  if (epsTarget > 0) then {
	     if ((epsTarget >= 2^(-53)) && (inputFormatA.accuracy <= 53) && (inputFormatB.accuracy <= 53)) then {
               res = {.ml_precision = "double", .accuracy = 53};

                "mll_computeOutputFormatAddition 0";
           res;
	     } else {
	       accuracy = max(inputFormatA.accuracy, inputFormatB.accuracy, ceil(max(2, ceil(-log2(epsTarget)) / 51) * 51));
           "mll_computeOutputFormatAddition 1";
           accuracy;
           res = mll_getFormatFromAccuracy(accuracy);
           res;
	     };
	  } else {
	     accuracy = max(inputFormatA.accuracy, inputFormatB.accuracy);
           "mll_computeOutputFormatAddition 2";
           accuracy;
         res = mll_getFormatFromAccuracy(accuracy);
         res;
	  };
	  
	  return res;
};

/* Rounds a given coefficient c into a format that guarantees
   that the rounding error is less than epsTarget. The function
   does not return the retained format but the rounded number.

   epsTarget is a positive OR ZERO number.

   If epsTarget is zero, the function is supposed to check
   whether there exists a format such that the constant can be
   represented exactly.

   The function returns a structure with at least two fields

   *   .okay indicating that the rounding was able to be performed
   *   .c    the rounded constant

*/
procedure mll_roundConstantToTargetPrecFormatSpecific(c, epsTarget) {
	  var res;
	  var cT;
	  var cR, cN;
	  var oldPrec;

	  res = { .okay = false };
	  if (epsTarget >= 0) then {
	     if (epsTarget == 0) then {
	     	if (c == 0) then {
		   res = { .okay = true, .c = c };
		} else {
		   oldPrec = prec;
		   prec = 10000!;
		   cT = [c];
		   prec = oldPrec!;
		   cR = mid(cT);
		   if (cR == c) then {
		      res = { .okay = true, .c = c };
		   };
		};
	     } else {
	        if (c == 0) then {
		   res = { .okay = true, .c = c };
		} else {
		   cR = round(c, D, RN);
		   while (abs(cR/c - 1) > epsTarget) do {
		   	 cN = round(c - cR, D, RN);
			 cR = cR + cN;
		   };
		   res = { .okay = true, .c = cR };
		};
	     };
	  };

	  return res;
};

/* This function takes a constant that has been obtained
   using the function mll_roundConstantToTargetPrecFormatSpecific
   and returns the format of this constant.

   The input is a real number.
   The output is of the "opaque format" type.
*/
procedure mll_computeFormatForConstant(c) {
	  var res, cR, cN;

	  if (c == 0) then {
	     res = {.ml_precision="double", .accuracy=53};
	  } else {
	     accuracy = 0;
	     cN = c;
	     while (cN != 0) do {
	     	   cR = round(cN, D, RN);
		   cN = cN - cR;
		   accuracy = accuracy + 53;
	     };
         "mll_computeFormatForConstant";
         c;
         accuracy;
         res = mll_getFormatFromAccuracy(accuracy);
         res;
	  };

	  return res;
};

/* This function computes an ordering on two opaque format variables.

   The function returns

   *     -1    if af provides less precision than bf
   *      0    if af and bf provide the same precision
   *      1    if af provides more precision that bf.

*/
procedure mll_compareFormats(af, bf) {
	  var res;

	  if (af.accuracy < bf.accuracy) then {
	     res = -1;
	  } else {
	     if (af.accuracy == bf.accuracy) then {
                res = 0;
	     } else {
	        res = 1;
	     };
	  };

	  return res;
};

/* The function computes the format of the _x_ variable, which
   comes as a variableFormat naturally, when it needs to be known
   up to a relative error epsTarget over I.
   
   I is an interval.
   epsTarget is a positive or zero real number.
   variableFormat is of type "opaque format type"

   The result is of type "opaque format type".

   When epsTarget is zero, it makes sense to return variableFormat
   as "rounding" a variable _x_ that is naturally stored as
   a variableFormat-variable to variableFormat does not imply any
   error.
   
*/
procedure mll_computeNeededVariableFormat(I, epsTarget, variableFormat) {
	  var res;

	  res = variableFormat;
	  if (epsTarget > 0) then {
	     if (epsTarget >= 2^(-53)) then {
             "mll_computeNeededVariableFormat 0";
	     	res = mll_getFormatFromAccuracy(53);
            res;
	     } else {
	        res = mll_getFormatFromAccuracy(ceil(max(2, ceil(-log2(epsTarget))) / 52) * 52);
             "mll_computeNeededVariableFormat 1";
            res;
             };
	  };

	  return res;
};

/* Returns a bound on the relative error implied by "rounding" _x_,
   which is naturally stored as an af-format variable,
   to a bf-format variable.

   If bf is "larger" (i.e. more precise) than af,
   the error is most probably zero as this just means
   "upconverting" a variable.

   The inputs af and bf are of the "opaque format type".

   The result is a positive or zero real number.

*/
procedure mll_computeErrorDueToVariableRounding(af, bf) {
	  var res;

	  if (af.accuracy <= bf.accuracy) then {
	     res = 0;
	  } else {
	     res = 2^(-bf.accuracy + 1);
	  };

	  return res;
};

/* Return a bound on the relative error for powering the _x_ variable by k,
   which is on a variableFormat format, producing the power result
   on an outputFormat format.

   The input types is an integer k of the power to compute and opaque format types.
   The output type is a positive or zero real number
*/
procedure mll_computeBoundPower(k, outputFormat, variableFormat) {
	var res;
	  res = 2^(-outputFormat.accuracy);
      return res;
};


/* Returns the output format of a variable _x_ powering by k, knowing
   that the _x_ variable is on a variableFormat format and we
   need to achieve an error of less than epsTarget.

   The input and output formats are of type "opaque format type".
   The epsTarget input is a positive or zero real number.
   The k input is an integer.

*/
procedure mll_computeOutputFormatPower(k, epsTarget, variableFormat) {
	  var accuracy;
      var res;
	  if (epsTarget > 0) then {
	     if (epsTarget >= 2^(-53)) then {
               accuracy = 53;
	     } else {
	       accuracy = ceil(max(2, ceil(-log2(epsTarget)) / 51) * 51);
	     };
	  };

	  
	  res = mll_getFormatFromAccuracy(accuracy);

      return res;
};

// inserting variable format type (which should depend on
// input rather than being static)
mll_format = {
    .ml_precision = "dd", // ML_DoubleDouble
    .accuracy = 102 // accuracy (upper bound on the relative error, in bits)
};


// instanciation of opaque type to determine Metalibm's lugdunum format
// (dummy implementation for now)
mll_context = { 
        // .variableFormat = mll_format,
	    .compareFormats = mll_compareFormats,
        // .computeBoundAddition = mll_computeActualAdditionBound,
        // .computeBoundMultiplication = mll_computeActualMultiplicationBound,
	    .computeBoundVariableRounding = mll_computeErrorDueToVariableRounding,
	    .computeOutputFormatAddition = mll_computeOutputFormatAddition,	
	    .computeOutputFormatMultiplication = mll_computeOutputFormatMultiplication,
	    .computeConstantFormat = mll_computeFormatForConstant,
	    .computeVariableFormat = mll_computeNeededVariableFormat,
	    // .computeBoundPower = mll_computeBoundPower,
	    .computeOutputFormatPower = mll_computeOutputFormatPower,
	    .roundConstant = mll_roundConstantToTargetPrecFormatSpecific	
};
