/* Helper procedure */
procedure __combineDomains(I, J) {
	  return [ max(inf(I), inf(J)); min(sup(I), sup(J)) ];
};

/* Helper procedures */
procedure __de_encloseZerosOfPolynomialInner(p, I) {
          var zz, z, zeros;

          zz = dirtyfindzeros(p, I);

          zeros = [||];
          for z in zz do {
              Z = z * [1 - 2^(-prec+1);1 + 2^(-prec+1)];
              if (p(inf(Z)) * p(sup(Z)) <= 0) then {
                 zeros = Z .: zeros;
              };
          };
          zeros = revert(zeros);

          return zeros;
};

procedure __de_encloseZerosOfPolynomialNoZero(p, I) {
          var nbZeros;
          var okay;
          var oldPrec, oldPoints;
          var zeros;

          nbZeros = numberroots(p, I);
          oldPrec = prec;
          oldPoints = points;
          zeros = [||];
          okay = (nbZeros == length(zeros));
          while (!okay) do {
                zeros = __de_encloseZerosOfPolynomialInner(p, I);
                okay = (nbZeros == length(zeros));
                if (!okay) then {
                   prec = ceil(prec * 1.25)!;
                   points = ceil(points * 1.25)!;
                };
          };
          prec = oldPrec!;
          points = oldPoints!;

          return zeros;
};

procedure __de_encloseZerosOfPolynomial(p, I) {
          var zeros;
          var pp, k;

          if (0 in I) then {
             pp = p;
             k = 0;
             while (mod(pp, _x_) == 0) do {
                   pp = div(pp, _x_);
                   k = k + 1;
             };
             if (horner(p - _x_^k * pp) == 0) then {
                zeros = [0] .: __de_encloseZerosOfPolynomialNoZero(pp, I);
             } else {
                zeros = __de_encloseZerosOfPolynomialNoZero(p, I);
             };
          } else {
             zeros = __de_encloseZerosOfPolynomialNoZero(p, I);
          };

          return zeros;
};

procedure __de_computeImagePolynomial(p, I) {
          var a, b, c, d, J, zeros, z;

          a = inf(I);
          b = sup(I);
          if (degree(p) > 1) then {
             zeros = [a] .: (__de_encloseZerosOfPolynomial(diff(p), I)) :. [b];
             J = evaluate(p, zeros[0]);
             c = inf(J);
             d = sup(J);
             for z in zeros do {
                  J = evaluate(p, z);
                  c = min(c, inf(J));
                  d = max(d, sup(J));
             };
          } else {
             J = evaluate(p, [a]);
             c = inf(J);
             d = sup(J);
             J = evaluate(p, [b]);
             c = min(c, inf(J));
             d = max(d, sup(J));
          };

          return [c; d];
};

/* Return a bound on the relative error for a multiplication
   multiplying an inputFormatA variable by an inputFormatB variable,
   returning an outputFormatVariable.

   The input types are opaque format types.
   The output type is a positive or zero real number
*/
procedure computeActualMultiplicationBound(outputFormat, inputFormatA, inputFormatB) {
	  return 2^(-(outputFormat.bits));
};

/* Return a bound on the absolute error for a multiplication
   multiplying an inputFormatA variable by an inputFormatB variable,
   returning an outputFormatVariable.

   The input types are opaque format types.
   The output type is a positive or zero real number
*/
procedure computeActualMultiplicationBoundAbsolute(outputFormat, inputFormatA, inputFormatB) {
	  var delta, eps, p, J, I;

	  eps = computeActualMultiplicationBound(outputFormat, inputFormatA, inputFormatB);

	  if (eps == 0) then {
	     delta = 0;
	  } else {
	     p = outputFormat.p;
	     I = outputFormat.I;
	     J = __de_computeImagePolynomial(p, I);
	     delta = abs(eps) * sup(abs(J));
	  };

	  return delta;
};

/* Return a bound on the relative error for an addition
   adding an inputFormatA variable by an inputFormatB variable,
   returning an outputFormatVariable.

   The input types are opaque format types.
   The output type is a positive or zero real number
*/
procedure computeActualAdditionBound(outputFormat, inputFormatA, inputFormatB) {
	  return 2^(-(outputFormat.bits));
};

/* Return a bound on the absolute error for an addition
   adding an inputFormatA variable by an inputFormatB variable,
   returning an outputFormatVariable.

   The input types are opaque format types.
   The output type is a positive or zero real number
*/
procedure computeActualAdditionBoundAbsolute(outputFormat, inputFormatA, inputFormatB) {
	  var delta, eps, p, J, I;

	  eps = computeActualAdditionBound(outputFormat, inputFormatA, inputFormatB);

	  if (eps == 0) then {
	     delta = 0;
	  } else {
	     p = outputFormat.p;
	     I = outputFormat.I;
	     J = __de_computeImagePolynomial(p, I);
	     delta = abs(eps) * sup(abs(J));
	  };

	  return delta;
};

/* Return a bound on the relative error for powering the _x_ variable by k,
   which is on a variableFormat format, producing the power result
   on an outputFormat format.

   The input types is an integer k of the power to compute and opaque format types.
   The output type is a positive or zero real number
*/
procedure computeBoundPower(k, outputFormat, variableFormat) {
	  return 2^(-(outputFormat.bits));
};

/* Return a bound on the absolute error for powering the _x_ variable by k,
   which is on a variableFormat format, producing the power result
   on an outputFormat format.

   The input types is an integer k of the power to compute and opaque format types.
   The output type is a positive or zero real number
*/
procedure computeBoundPowerAbsolute(k, outputFormat, variableFormat) {
	  var delta, eps, p, J, I;

	  eps = computeBoundPower(k, outputFormat, variableFormat);

	  if (eps == 0) then {
	     delta = 0;
	  } else {
	     p = outputFormat.p;
	     I = outputFormat.I;
	     J = __de_computeImagePolynomial(p, I);
	     delta = abs(eps) * sup(abs(J));
	  };

	  return delta;
};

/* Returns the output format of a multiplication that will be fed two operands
   of formats inputFormatA and inputFormatB and that is supposed to have a
   relative error less than or equal to epsTarget.

   The input and output formats are of type "opaque format type".
   The epsTarget input is a positive or zero real number.

*/
procedure computeOutputFormatMultiplication(epsTarget, inputFormatA, inputFormatB) {
	  var res, bits;

	  if (epsTarget > 0) then {
	     if ((epsTarget >= 2^(-53)) && (inputFormatA.bits <= 53) && (inputFormatB.bits <= 53)) then {
               bits = 53;
	     } else {
	       bits = max(inputFormatA.bits, inputFormatB.bits, ceil(max(2, ceil(-log2(epsTarget)) / 51) * 51));
	     };
	  } else {
	     bits = max(inputFormatA.bits, inputFormatB.bits);
	  };

	  res = { .bits = bits, .p = inputFormatA.p * inputFormatB.p, .I = __combineDomains(inputFormatA.I, inputFormatB.I) };
	  
	  return res;
};

/* Returns the output format of a multiplication that will be fed two operands
   of formats inputFormatA and inputFormatB and that is supposed to have a
   absolute error less than or equal to deltaTarget.

   The input and output formats are of type "opaque format type".
   The deltaTarget input is a positive or zero real number.

*/
procedure computeOutputFormatMultiplicationAbsolute(deltaTarget, inputFormatA, inputFormatB) {
	  var res, epsTarget, p, I, J;

	  if (deltaTarget == 0) then {
	     epsTarget = 0;
	  } else {
	     p = inputFormatA.p * inputFormatB.p;
	     I = __combineDomains(inputFormatA.I, inputFormatB.I);
	     J = __de_computeImagePolynomial(p, I);
	     if (sup(abs(J)) == 0) then {
	     	epsTarget = 1/2;
	     } else {
	       	epsTarget = min(1/2, abs(deltaTarget) / sup(abs(J)));
	     };
	  };

	  res = computeOutputFormatMultiplication(epsTarget, inputFormatA, inputFormatB);
	  
	  return res;
};

/* Returns the output format of an addition that will be fed two operands
   of formats inputFormatA and inputFormatB and that is supposed to have a
   relative error less than or equal to epsTarget.

   The input and output formats are of type "opaque format type".
   The epsTarget input is a positive or zero real number.

*/
procedure computeOutputFormatAddition(epsTarget, inputFormatA, inputFormatB) {
	  var res, bits;

	  if (epsTarget > 0) then {
	     if ((epsTarget >= 2^(-53)) && (inputFormatA.bits <= 53) && (inputFormatB.bits <= 53)) then {
               bits = 53;
	     } else {
	       bits = max(inputFormatA.bits, inputFormatB.bits, ceil(max(2, ceil(-log2(epsTarget)) / 51) * 51));
	     };
	  } else {
	     bits = max(inputFormatA.bits, inputFormatB.bits);
	  };

	  res = { .bits = bits, .p = inputFormatA.p + inputFormatB.p, .I = __combineDomains(inputFormatA.I, inputFormatB.I) };
	  
	  return res;
};

/* Returns the output format of an addition that will be fed two operands
   of formats inputFormatA and inputFormatB and that is supposed to have a
   absolute error less than or equal to deltaTarget.

   The input and output formats are of type "opaque format type".
   The deltaTarget input is a positive or zero real number.

*/
procedure computeOutputFormatAdditionAbsolute(deltaTarget, inputFormatA, inputFormatB) {
	  var res, epsTarget, p, I, J;

	  if (deltaTarget == 0) then {
	     epsTarget = 0;
	  } else {
	     p = inputFormatA.p + inputFormatB.p;
	     I = __combineDomains(inputFormatA.I, inputFormatB.I);
	     J = __de_computeImagePolynomial(p, I);
	     if (sup(abs(J)) == 0) then {
	     	epsTarget = 1/2;
	     } else {
	       	epsTarget = min(1/2, abs(deltaTarget) / sup(abs(J)));
	     };
	  };

	  res = computeOutputFormatAddition(epsTarget, inputFormatA, inputFormatB);
	  
	  return res;
};


/* Returns the output format of a variable _x_ powering by k, knowing
   that the _x_ variable is on a variableFormat format and we
   need to achieve a relative error of less than epsTarget.

   The input and output formats are of type "opaque format type".
   The epsTarget input is a positive or zero real number.
   The k input is an integer.

*/
procedure computeOutputFormatPower(k, epsTarget, variableFormat) {
	  var res, bits;

	  if (epsTarget > 0) then {
	     if (epsTarget >= 2^(-53)) then {
               bits = 53;
	     } else {
	       bits = ceil(max(2, ceil(-log2(epsTarget)) / 51) * 51);
	     };
	  };

	  res = { .bits = bits, .p = (variableFormat.p)^k, .I = variableFormat.I };

	  return res;
};

/* Returns the output format of a variable _x_ powering by k, knowing
   that the _x_ variable is on a variableFormat format and we
   need to achieve an absolute error of less than deltaTarget.

   The input and output formats are of type "opaque format type".
   The deltaTarget input is a positive or zero real number.
   The k input is an integer.

*/
procedure computeOutputFormatPowerAbsolute(k, deltaTarget, variableFormat) {
	  var res, epsTarget, p, I, J;

	  if (deltaTarget == 0) then {
	     epsTarget = 0;
	  } else {
	     p = (variableFormat.p)^k;
	     I = variableFormat.I;
	     J = __de_computeImagePolynomial(p, I);
	     if (sup(abs(J)) == 0) then {
	     	epsTarget = 1/2;
	     } else {
	       	epsTarget = min(1/2, abs(deltaTarget) / sup(abs(J)));
	     };
	  };

	  res = computeOutputFormatPower(k, epsTarget, variableFormat);
	  
	  return res;
};

/* Rounds a given coefficient c into a format that guarantees that the
   relative rounding error is less than epsTarget. The function does
   not return the retained format but the rounded number.

   epsTarget is a positive OR ZERO number.

   If epsTarget is zero, the function is supposed to check
   whether there exists a format such that the constant can be
   represented exactly.

   The function returns a structure with at least two fields

   *   .okay indicating that the rounding was able to be performed
   *   .c    the rounded constant

*/
procedure roundConstantToTargetPrecFormatSpecific(c, epsTarget) {
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
   using the function roundConstantToTargetPrecFormatSpecific
   and returns the format of this constant.

   The input is a real number.
   The output is of the "opaque format" type.
*/
procedure computeFormatForConstant(c) {
	  var res, bits, cR, cN;

	  if (c == 0) then {
	     bits = 53;
	  } else {
	     bits = 0;
	     cN = c;
	     while (cN != 0) do {
	     	   cR = round(cN, D, RN);
		   cN = cN - cR;
		   bits = bits + 53;
	     };
	  };

	  res = { .bits = bits, .p = c, .I = [-infty; infty] };

	  return res;
};

/* This function computes an ordering on two opaque format variables.

   The function returns

   *     -1    if af provides less precision than bf
   *      0    if af and bf provide the same precision
   *      1    if af provides more precision that bf.

*/
procedure compareFormats(af, bf) {
	  var res;

	  if (af.bits < bf.bits) then {
	     res = -1;
	  } else {
	     if (af.bits == bf.bits) then {
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
procedure computeNeededVariableFormat(I, epsTarget, variableFormat) {
	  var res, bits;

	  bits = variableFormat.bits;
	  if (epsTarget > 0) then {
	     if (epsTarget >= 2^(-53)) then {
	     	bits = 53;
	     } else {
	        bits = ceil(max(2, ceil(-log2(epsTarget))) / 52) * 52;
             };
	  };

	  res = { .bits = bits, .p = variableFormat.p, .I = __combineDomains(I, variableFormat.I) };

	  return res;
};

/* The function computes the format of the _x_ variable, which
   comes as a variableFormat naturally, when it needs to be known
   up to an absolute error deltaTarget over I.
   
   I is an interval.
   deltaTarget is a positive or zero real number.
   variableFormat is of type "opaque format type"

   The result is of type "opaque format type".

   When deltaTarget is zero, it makes sense to return variableFormat
   as "rounding" a variable _x_ that is naturally stored as
   a variableFormat-variable to variableFormat does not imply any
   error.
   
*/
procedure computeNeededVariableFormatAbsolute(I, deltaTarget, variableFormat) {
	  var res, epsTarget, p, I, J;

	  if (deltaTarget == 0) then {
	     epsTarget = 0;
	  } else {
	     p = variableFormat.p;
	     I = __combineDomains(I, variableFormat.I);
	     J = __de_computeImagePolynomial(p, I);
	     if (sup(abs(J)) == 0) then {
	     	epsTarget = 1/2;
	     } else {
	       	epsTarget = min(1/2, abs(deltaTarget) / sup(abs(J)));
	     };
	  };

	  res = computeNeededVariableFormat(I, epsTarget, variableFormat);
	  
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
procedure computeErrorDueToVariableRounding(af, bf) {
	  var res;

	  if (af.bits <= bf.bits) then {
	     res = 0;
	  } else {
	     res = 2^(-(bf.bits) + 1);
	  };

	  return res;
};

/* Returns a bound on the absolute error implied by "rounding" _x_,
   which is naturally stored as an af-format variable,
   to a bf-format variable.

   If bf is "larger" (i.e. more precise) than af,
   the error is most probably zero as this just means
   "upconverting" a variable.

   The inputs af and bf are of the "opaque format type".

   The result is a positive or zero real number.

*/
procedure computeErrorDueToVariableRoundingAbsolute(af, bf) {
	  var delta, eps, p, J, I;

	  eps = computeErrorDueToVariableRounding(af, bf);

	  if (eps == 0) then {
	     delta = 0;
	  } else {
	     p = af.p;
	     I = bf.I;
	     J = __de_computeImagePolynomial(p, I);
	     delta = abs(eps) * sup(abs(J));
	  };

	  return delta;
};

