
suppressmessage(171, 457, 432);

procedure roundConstantToTargetPrec(c, rawEpsTarget, context) {
          var res;
          var epsTarget;

          if (c == 0) then {
             res = { .okay = true, .c = c };
          } else {
              epsTarget = abs(rawEpsTarget);
              if (epsTarget == infty) then {
                 epsTarget = 1b1024;
              };

              res = context.roundConstant(c, 0);
              if (!res.okay) then {
                 r = context.roundConstant(round(c, TD, RN), 0);
                 if (r.okay) then {
                    if (abs(r.c/c-1) < abs(epsTarget)) then {
                       res = r;
                    } else {
                       res = context.roundConstant(c, epsTarget);
                    };
                 } else {
                    res = context.roundConstant(c, epsTarget);
                 };
              };
          };

          return res;
};

procedure computeErrorBoundsUnifyErrorBound(eps) {
          var res, s, absEps;

          if (eps == eps) then {
              if (eps == 0) then {
                 res = eps;
              } else {
                if (eps < 0) then {
                   s = -1;
                   absEps = -eps;
                } else {
                   s = 1;
                   absEps = eps;
                };
                res = s * 2^(-floor(-log2(absEps)));
              };
          } else {
            res = infty;
          };

          return res;
};

procedure __computeErrorBoundsConstant(c, I, epsTarget, context) {
          var res;
          var r;
          var P;
          var eps;
          var fmt;

          res = { .okay = false, .reason = { .text = "Could not round constant", .data = { .c = c, .epsTarget = epsTarget } } };
          r = roundConstantToTargetPrec(c, epsTarget, context);
          if (r.okay) then {
             if (c == r.c) then {
                eps = 0;
             } else {
                eps = abs(r.c/c-1);
             };
             eps = computeErrorBoundsUnifyErrorBound(eps);
             fmt = context.computeConstantFormat(r.c);
             if (fmt == fmt) then {
                P = { .type = "constant", .c = r.c, .p = r.c, .format = fmt, .epsOp = 0, .epsOpTarget = epsTarget, .eps = round(eps,prec,RU), .delta = round(abs(eps * r.c),prec,RU), .image = [r.c,r.c], .domain = I };
                res = { .okay = true, .poly = P };
             };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectConstant(c, I, deltaTarget, context) {
          var res, epsTarget, r;

          res = { .okay = false, .reason = { .text = "Could not round constant", .data = { .c = c, .deltaTarget = deltaTarget } } };

	  if (c == 0) then {
	     epsTarget = 1/2;
	  } else {
	     epsTarget = min(1/2, abs(deltaTarget) / sup(abs(evaluate(c, 1))));
	     if (!((epsTarget == epsTarget) && (epsTarget >= 0))) then {
	     	epsTarget = 0;
	     };
	  };
	  r = __computeErrorBoundsConstant(c, I, epsTarget, context);
	  if (r.okay) then {
	     res = r;
	  };	  

          return res;
};

procedure __computeErrorBoundsVariable(I, epsTarget, context) {
          var res, P, neededVariableFormat, eps, img, cmp, delta;

          res = { .okay = false, .reason = { .text = "Could not handle variable", .data = { .I = I, .domain = I, .epsTarget = epsTarget } } };
          if (epsTarget >= 0) then {
             if (epsTarget == 0) then {
                P = { .type = "variable", .p = _x_, .format = context.variableFormat, .inputformat = context.variableFormat, .epsOp = 0, .epsOpTarget = epsTarget, .eps = 0, .delta = 0, .image = I, .domain = I };
                res = { .okay = true, .poly = P };
             } else {
                neededVariableFormat = context.computeVariableFormat(I, epsTarget, context.variableFormat);
                if (neededVariableFormat == neededVariableFormat) then {
                    cmp = context.compareFormats(neededVariableFormat, context.variableFormat);
                    if (cmp == cmp) then {
                       if (cmp < 0) then {
                          eps = computeErrorBoundsUnifyErrorBound(context.computeBoundVariableRounding(context.variableFormat, neededVariableFormat));
                          img = I * [1-eps;1+eps];
			  delta = sup(abs(I * eps));
                          P = { .type = "variable", .p = _x_, .format = neededVariableFormat, .inputformat = context.variableFormat, .epsOp = eps, .epsOpTarget = epsTarget, .eps = eps, .delta = delta, .image = img, .domain = I };
                          res = { .okay = true, .poly = P };
                       } else {
                          P = { .type = "variable", .p = _x_, .format = context.variableFormat, .inputformat = context.variableFormat, .epsOp = 0, .epsOpTarget = epsTarget, .eps = 0, .delta = 0, .image = I, .domain = I };
                          res = { .okay = true, .poly = P };
                       };
                    };
                };
             };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectVariableDirect(I, deltaTarget, context) {
          var res, P, neededVariableFormat, delta, img, cmp, eps, epsTarget;

          res = { .okay = false, .reason = { .text = "Could not handle variable", .data = { .I = I, .domain = I, .deltaTarget = deltaTarget } } };
          if (deltaTarget >= 0) then {
             if (deltaTarget == 0) then {
                P = { .type = "variable", .p = _x_, .format = context.variableFormat, .inputformat = context.variableFormat, .epsOp = 0, .epsOpTarget = 0, .eps = 0, .delta = 0, .deltaOp = 0, .deltaOpTarget = deltaTarget, .image = I, .domain = I };
                res = { .okay = true, .poly = P };
             } else {
                neededVariableFormat = context.computeVariableFormatAbsolute(I, deltaTarget, context.variableFormat);
                if (neededVariableFormat == neededVariableFormat) then {
                    cmp = context.compareFormats(neededVariableFormat, context.variableFormat);
                    if (cmp == cmp) then {
                       if (cmp < 0) then {
                          delta = computeErrorBoundsUnifyErrorBound(context.computeBoundVariableRoundingAbsolute(context.variableFormat, neededVariableFormat));
                          img = I * [-abs(delta);abs(delta)];
			  if (0 in I) then {
                             P = { .type = "variable", .p = _x_, .format = neededVariableFormat, .inputformat = context.variableFormat, .deltaOp = delta, .deltaOpTarget = deltaTarget, .delta = delta, .image = img, .domain = I };
			  } else {
			     eps = abs(delta) / inf(abs(I));
			     epsTarget = abs(deltaTarget) / sup(abs(I));
			     P = { .type = "variable", .p = _x_, .format = neededVariableFormat, .inputformat = context.variableFormat, .epsOp = eps, .deltaOp = delta, .epsOpTarget = epsTarget, .deltaOpTarget = deltaTarget, .eps = eps, .delta = delta, .image = img, .domain = I };
			  };
                          res = { .okay = true, .poly = P };
                       } else {
		          P = { .type = "variable", .p = _x_, .format = context.variableFormat, .inputformat = context.variableFormat, .epsOp = 0, .eps = 0, .delta = 0, .deltaOp = 0, .deltaOpTarget = deltaTarget, .image = I, .domain = I };
                          res = { .okay = true, .poly = P };
                       };
                    };
                };
             };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectVariableIndirect(I, deltaTarget, context) {
          var res, epsTarget, r;

          res = { .okay = false, .reason = { .text = "Could not handle variable", .data = { .I = I, .domain = I, .deltaTarget = deltaTarget } } };

	  epsTarget = min(1/2, abs(deltaTarget) / sup(abs(I)));
	  if ((epsTarget == epsTarget) && (epsTarget >= 0)) then {
	     r = __computeErrorBoundsVariable(I, epsTarget, context);
	     if (r.okay) then {
	     	res = r;
	     };
	  };

	  return res;
};

procedure __computeErrorBoundsAbsoluteDirectVariable(I, deltaTarget, context) {
          var res, r;

          res = { .okay = false, .reason = { .text = "Could not handle variable", .data = { .I = I, .domain = I, .deltaTarget = deltaTarget } } };

	  match (context) with
	  	{ .computeVariableFormatAbsolute = computeVariableFormatAbsolute,
		  .computeBoundVariableRoundingAbsolute = computeBoundVariableRoundingAbsolute } : {
												r = __computeErrorBoundsAbsoluteDirectVariableDirect(I, deltaTarget, context);
		  					  			       	   }
		default : {
												r = __computeErrorBoundsAbsoluteDirectVariableIndirect(I, deltaTarget, context);
			  };

	  if (r.okay) then {
	     res = r;
	  };

	  return res;
};

procedure __computeErrorBoundsPower(k, I, epsTarget, context) {
          var res, p, P, img, fmt, actualEps, delta, II, tmp;

          res = { .okay = false, .reason = { .text = "Could not handle powering", .data = { .k = k, .I = I, .domain = I, .epsTarget = epsTarget } } };
          if ((epsTarget >= 0) && (k >= 2)) then {
	     if (epsTarget == 0) then {
	     	tmp = 0;
	     } else {
	        tmp = 2^(-ceil(-log2(abs(epsTarget))));
	     };
             fmt = context.computeOutputFormatPower(k, tmp, context.variableFormat);
             if (fmt == fmt) then {
                 actualEps = computeErrorBoundsUnifyErrorBound(context.computeBoundPower(k, fmt, context.variableFormat));
                 if (actualEps <= epsTarget) then {
                    p = _x_^k;
		    II = computeImagePolynomial(p, I);
                    img = II * [1 - actualEps; 1 + actualEps];
		    delta = sup(abs(II * actualEps));
                    P = { .type = "power", .k = k, .p = p, .format = fmt, .inputformat = context.variableFormat, .epsOp = actualEps, .epsOpTarget = epsTarget, .eps = actualEps, .delta = delta, .image = img, .domain = I };
                    res = { .okay = true, .poly = P };
                 };
              };
          };
          return res;
};

procedure __computeErrorBoundsAbsoluteDirectPowerIndirect(k, I, deltaTarget, context) {
          var res, epsTarget, r;

          res = { .okay = false, .reason = { .text = "Could not handle powering", .data = { .k = k, .I = I, .domain = I, .deltaTarget = deltaTarget } } };

	  epsTarget = min(1/2, abs(deltaTarget) / sup(abs(I)));
	  if ((epsTarget == epsTarget) && (epsTarget >= 0)) then {
	     r = __computeErrorBoundsPower(k, I, epsTarget, context);
	     if (r.okay) then {
	     	res = r;
	     };
	  };

	  return res;
};

procedure __computeErrorBoundsAbsoluteDirectPowerDirect(k, I, deltaTarget, context) {
          var res, p, P, img, fmt, actualDelta, II, tmp, actualEps, epsTarget;

          res = { .okay = false, .reason = { .text = "Could not handle powering", .data = { .k = k, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
          if ((deltaTarget >= 0) && (k >= 2)) then {
	     if (deltaTarget == 0) then {
	     	tmp = 0;
	     } else {
	        tmp = 2^(-ceil(-log2(abs(deltaTarget))));
	     };
             fmt = context.computeOutputFormatPowerAbsolute(k, tmp, context.variableFormat);
             if (fmt == fmt) then {
                 actualDelta = computeErrorBoundsUnifyErrorBound(context.computeBoundPowerAbsolute(k, fmt, context.variableFormat));
                 if (actualDelta <= deltaTarget) then {
                    p = _x_^k;
		    II = computeImagePolynomial(p, I);
                    img = II + [-abs(actualDelta);abs(actualDelta)];
		    if (0 in II) then {
                       P = { .type = "power", .k = k, .p = p, .format = fmt, .inputformat = context.variableFormat, .delta = actualDelta, .deltaOp = actualDelta, .deltaOpTarget = deltaTarget, .image = img, .domain = I };
		    } else {
		       actualEps = abs(actualDelta) / inf(abs(II));
		       epsTarget = abs(deltaTarget) / sup(abs(II));
                       P = { .type = "power", .k = k, .p = p, .format = fmt, .inputformat = context.variableFormat, .epsOp = actualEps, .epsOpTarget = epsTarget, .eps = actualEps, .delta = actualDelta, .deltaOp = actualDelta, .deltaOpTarget = deltaTarget, .image = img, .domain = I };
		    };
                    res = { .okay = true, .poly = P };
                 };
              };
          };
          return res;
};

procedure __computeErrorBoundsAbsoluteDirectPower(k, I, deltaTarget, context) {
          var res, r;

          res = { .okay = false, .reason = { .text = "Could not handle powering", .data = { .k = k, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
	  
	  match (context) with
	  	{ .computeOutputFormatPowerAbsolute = computeOutputFormatPowerAbsolute,
		  .computeBoundPowerAbsolute = computeBoundPowerAbsolute  } : {
												r = __computeErrorBoundsAbsoluteDirectPowerDirect(k, I, deltaTarget, context);
		  					  			       	   }
		default : {
												r = __computeErrorBoundsAbsoluteDirectPowerIndirect(k, I, deltaTarget, context);
			  };

	  if (r.okay) then {
	     res = r;
	  };


	  return res;
};


procedure __encloseZerosOfPolynomialInner(p, I) {
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

procedure __encloseZerosOfPolynomialNoZero(p, I) {
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
                zeros = __encloseZerosOfPolynomialInner(p, I);
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

procedure encloseZerosOfPolynomial(p, I) {
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
                zeros = [0] .: __encloseZerosOfPolynomialNoZero(pp, I);
             } else {
                zeros = __encloseZerosOfPolynomialNoZero(p, I);
             };
          } else {
             zeros = __encloseZerosOfPolynomialNoZero(p, I);
          };

          return zeros;
};

procedure computeImagePolynomial(p, I) {
          var a, b, c, d, J, zeros, z;

          a = inf(I);
          b = sup(I);
          if (degree(p) > 1) then {
             zeros = [a] .: (encloseZerosOfPolynomial(diff(p), I)) :. [b];
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

procedure computeImageRationalFunction(p, q, I) {
          var f, g, t, h;
          var res;
	  var a, b, c, d;
	  var nbZerosG, w, zeros, J, z;

          t = gcd(p, q);
          f = div(p, t);
          g = div(q, t);
          if ((degree(g) == 0) || (mod(f, g) == 0)) then {
             if (mod(f, g) == 0) then {
                h = div(f, g);
             } else {
                h = f / g;
             };
             res = computeImagePolynomial(h, I);
          } else {
             nbZerosG = numberroots(g, I);
             if (nbZerosG != 0) then {
                res = [-infty; infty];
             } else {
                a = inf(I);
                b = sup(I);
                w = g * diff(f) - f * diff(g);
                zeros = [a] .: (encloseZerosOfPolynomial(w, I)) :. [b];
                J = evaluate(f/g, zeros[0]);
                c = inf(J);
                d = sup(J);
                for z in zeros do {
                    J = evaluate(f/g, z);
                    c = min(c, inf(J));
                    d = max(d, sup(J));
                };
                res = [c;d];
             };
          };

          return res;
};

procedure __hasEps(T) {
	  var res;

	  if (T.okay) then {
	     match (T.poly) with
	     	   { .eps = eps } : {
					res = true;
			            }
		   default :        {
					res = false;
		   	   	    };
	  } else {
	     res = false;
	  };

	  return res;
};

procedure __computeErrorBoundsMultiplicationDoit(q, r, p, ops, I, epsTarget, context, switchToAbsolute) {
          var res, P, okay, hopeless, beta;
          var epsMul, epsR, epsQ;
          var actualEpsMul, actualEpsQ, actualEpsR, actualEps;
          var actualPolynomial, img;
          var Q, R;
          var fmt;
	  var delta, II;

          res = { .okay = false, .reason = { .text = "Could not handle multiplication", .data = { .q = q, .r = r, .ops = ops, .p = p, .I = I, .domain = I, .epsTarget = epsTarget } } };
          okay = false;
          hopeless = false;
          beta = 1;
          while ((!okay) && (!hopeless)) do {
                epsMul = inf([(1/ops) * beta * epsTarget]);
                epsR = ((r.ops)/ops) * beta * epsTarget * 1/(1 + epsMul);
                epsQ = ((q.ops)/ops) * beta * epsTarget * 1/((1 + epsMul) * (1 + epsR));
                Q = __computeErrorBoundsRecurse(q, I, epsQ, context, switchToAbsolute);
                if (Q.okay && __hasEps(Q)) then {
                   R = __computeErrorBoundsRecurse(r, I, epsR, context, switchToAbsolute);
                   if (R.okay && __hasEps(R)) then {
                      actualEpsQ = Q.poly.eps;
                      actualEpsR = R.poly.eps;
                      fmt = context.computeOutputFormatMultiplication(epsMul, Q.poly.format, R.poly.format);
                      if (fmt == fmt) then {
                          actualEpsMul = computeErrorBoundsUnifyErrorBound(context.computeBoundMultiplication(fmt, Q.poly.format, R.poly.format));
                          actualEps = round(actualEpsQ * (1 + actualEpsR) * (1 + actualEpsMul) + actualEpsR * (1 + actualEpsMul) + actualEpsMul,prec,RU);
                          if ((actualEps == actualEps) && (actualEps <= epsTarget)) then {
                             actualPolynomial = Q.poly.p * R.poly.p;
			     II = computeImagePolynomial(actualPolynomial, I);
                             img = II * [1 - actualEps; 1 + actualEps];
			     delta = sup(abs(II * actualEps));
                             P = { .type = "multiplication", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .epsOp = actualEpsMul, .epsOpTarget = epsMul, .eps = actualEps, .delta = delta, .image = img, .domain = I };
                             okay = true;
                          };
                      } else {
                          hopeless = true;
                      };
                   } else {
                      hopeless = true;
                   };
                } else {
                   hopeless = true;
                };
                if ((!okay) && (!hopeless)) then {
                   beta = beta / 1.25;
                };
          };
          if (okay) then {
             res = { .okay = true, .poly = P };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectMultiplicationDoitIndirect(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
          var res, epsTarget, r;

          res = { .okay = false, .reason = { .text = "Could not handle multiplication", .data = { .q = q, .r = r, .ops = ops, .p = p, .I = I, .domain = I, .deltaTarget = deltaTarget } } };

	  if (switchToRelative) then {
	     epsTarget = min(1/2, abs(deltaTarget) / sup(abs(I)));
	     if ((epsTarget == epsTarget) && (epsTarget >= 0)) then {
	     	r = __computeErrorBoundsMultiplicationDoit(q, r, p, ops, I, epsTarget, context, false);
	     	if (r.okay) then {
	     	   res = r;
	     	};
	     };
	  };

	  return res;
};

procedure __computeErrorBoundsAbsoluteDirectMultiplicationDoitDirect(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
          var res, P, okay, hopeless, beta;
          var epsMul, deltaR, deltaQ, deltaMul;
          var actualEpsMul, actualDeltaMul, actualDeltaQ, actualDeltaR, actualDelta;
          var actualEpsQ, actualEpsR;
          var actualPolynomial, img;
          var Q, R;
          var fmt;
          var actualEps;
	  var alphaQ, alphaR, etaR, etaQ;
	  var actualAlphaQ, actualAlphaR;

          res = { .okay = false, .reason = { .text = "Could not handle multiplication", .data = { .q = q, .r = r, .p = p, .ops = ops, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
          okay = false;
          hopeless = false;
          beta = 1;
          while ((!okay) && (!hopeless)) do {
	  	alphaQ = abs(sup(abs(computeImagePolynomial(q.p, I))));
	  	alphaR = abs(sup(abs(computeImagePolynomial(r.p, I))));
	  	etaR = alphaQ;
                deltaR = ((r.ops)/ops) * beta * deltaTarget * 1/etaR;
		etaQ = alphaR + deltaR;
                deltaQ = ((q.ops)/ops) * beta * deltaTarget * 1/etaQ;
                deltaMul = (1/ops) * beta * deltaTarget;
                epsMul = round(deltaMul / sup(abs(computeImagePolynomial(p, I))),prec,RD);
                Q = __computeErrorBoundsAbsoluteDirectRecurse(q, I, deltaQ, context, switchToRelative);
                if (Q.okay) then {
                   R = __computeErrorBoundsAbsoluteDirectRecurse(r, I, deltaR, context, switchToRelative);
                   if (R.okay) then {
                      actualEpsQ = (match (Q.poly) with
		      		   	  { .eps = _eps } : (Q.poly.eps)
				 	  default :         (infty));
		      actualEpsR = (match (R.poly) with
		      		   	  { .eps = _eps } : (R.poly.eps)
				 	  default :         (infty));
                      match (Q.poly) with
                            { .delta = Qdelta } : { actualDeltaQ = Qdelta; }
                            default : {
                                        actualDeltaQ = sup(abs(computeImagePolynomial(Q.poly.p, I))) * actualEpsQ;
                                      };
                      match (R.poly) with
                            { .delta = Rdelta } : { actualDeltaR = Rdelta; }
                            default : {
                                        actualDeltaR = sup(abs(computeImagePolynomial(R.poly.p, I))) * actualEpsR;
                                      };
		      fmt = context.computeOutputFormatMultiplicationAbsolute(deltaMul,
									      Q.poly.format, R.poly.format);
                      if (fmt == fmt) then {
                          actualPolynomial = Q.poly.p * R.poly.p;
			  actualDeltaMul = computeErrorBoundsUnifyErrorBound(context.computeBoundMultiplicationAbsolute(fmt, Q.poly.format, R.poly.format));
                          actualEpsMul = round(abs(actualDeltaMul / inf(abs(computeImagePolynomial(actualPolynomial, I)))),prec,RU);
	  	          actualAlphaQ = abs(sup(abs(computeImagePolynomial(Q.poly.p, I))));
	  	          actualAlphaR = abs(sup(abs(computeImagePolynomial(R.poly.p, I))));
                          actualDelta = round(actualAlphaQ * actualDeltaR + actualAlphaR * actualDeltaQ + actualDeltaQ * actualDeltaR + actualDeltaMul,prec,RU);
                          actualEps = round(actualDelta / inf(abs(computeImagePolynomial(actualPolynomial, I))),prec,RU);
                          if ((actualDelta == actualDelta) && (actualDelta <= deltaTarget)) then {
                             img = computeImagePolynomial(actualPolynomial, I) + [-actualDelta; +actualDelta];
			     if ((actualEpsMul == actualEpsMul) && (actualEpsMul >= 0) && (actualEpsMul != infty) &&
			         (epsMul == epsMul) && (epsMul >= 0) && (epsMul != infty) &&
				 (actualEps == actualEps) && (actualEps >= 0) && (actualEps != infty)) then {
                             	P = { .type = "multiplication", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .epsOp = actualEpsMul, .epsOpTarget = epsMul, .eps = actualEps, .delta = actualDelta, .deltaOp = actualDeltaMul, .deltaOpTarget = deltaMul, .image = img, .domain = I };
		             } else {
                             	P = { .type = "multiplication", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .delta = actualDelta, .deltaOp = actualDeltaMul, .deltaOpTarget = deltaMul, .image = img, .domain = I };
			     };
                             okay = true;
                          };
                      } else {
                        hopeless = true;
                      };
                   } else {
                      hopeless = true;
                   };
                } else {
                   hopeless = true;
                };
                if ((!okay) && (!hopeless)) then {
                   beta = beta / 1.25;
                };
          };
          if (okay) then {
             res = { .okay = true, .poly = P };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectMultiplicationDoit(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
          var res, r;

          res = { .okay = false, .reason = { .text = "Could not handle multiplication", .data = { .q = q, .r = r, .ops = ops, .p = p, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
	  
	  match (context) with
	  	{ .computeOutputFormatMultiplicationAbsolute = computeOutputFormatMultiplicationAbsolute,
		  .computeBoundMultiplicationAbsolute = computeBoundMultiplicationAbsolute  } : {
												r = __computeErrorBoundsAbsoluteDirectMultiplicationDoitDirect(q, r, p, ops, I, deltaTarget, context, switchToRelative);
		  					  			       	   }
		default : {
												r = __computeErrorBoundsAbsoluteDirectMultiplicationDoitIndirect(q, r, p, ops, I, deltaTarget, context, switchToRelative);
			  };

	  if (r.okay) then {
	     res = r;
	  };

          return res;
};

procedure __computeErrorBoundsMultiplication(q, r, p, ops, I, epsTarget, context, switchToAbsolute) {
	  var res;

	  res = __computeErrorBoundsMultiplicationDoit(q, r, p, ops, I, epsTarget, context, switchToAbsolute);

	  return res;
};

procedure __computeErrorBoundsAbsoluteDirectMultiplication(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
	  var res;

	  res = __computeErrorBoundsAbsoluteDirectMultiplicationDoit(q, r, p, ops, I, deltaTarget, context, switchToRelative);

	  return res;
};

procedure __computeNoCancellationPossibleSign(q, r, I, epsQ, epsR) {
          var res, QRange, RRange;

          res = false;

          QRange = computeImagePolynomial(q, I) * [1 - epsQ; 1 + epsQ];
          RRange = computeImagePolynomial(r, I) * [1 - epsR; 1 + epsR];

          if ((!(0 in QRange)) && (!(0 in RRange))) then {
             if (mid(QRange) * mid(RRange) > 0) then {
                res = true;
             };
          };

          return res;
};

procedure __computeNoCancellationPossibleRatio(q, r, I, epsQ, epsR) {
          var res;
          var ratioBound;

          res = false;
          if (degree(q) < degree(r)) then {
             ratioBound = computeImageRationalFunction(r, q, I) * [ (1 - epsR) / (1 + epsQ) ; (1 + epsR) / (1 - epsQ) ];
             if (sup(abs(ratioBound)) == infty) then {
                ratioBound = computeImageRationalFunction(q, r, I) * [ (1 - epsQ) / (1 + epsR) ; (1 + epsQ) / (1 - epsR) ];
             };
          } else {
             ratioBound = computeImageRationalFunction(q, r, I) * [ (1 - epsQ) / (1 + epsR) ; (1 + epsQ) / (1 - epsR) ];
             if (sup(abs(ratioBound)) == infty) then {
                ratioBound = computeImageRationalFunction(r, q, I) * [ (1 - epsR) / (1 + epsQ) ; (1 + epsR) / (1 - epsQ) ];
             };
          };

          if (ratioBound in [-1/2;1/2]) then {
             res = true;
          } else {
            if ((inf(ratioBound) > 2) ||
                (sup(ratioBound) < -2)) then {
                res = true;
            };
          };

          return res;
};

procedure computeNoCancellationPossible(q, r, I, epsQ, epsR) {
          var res;

          res = false;
          if (__computeNoCancellationPossibleSign(q, r, I, epsQ, epsR)) then {
            res = true;
          } else {
            if (__computeNoCancellationPossibleRatio(q, r, I, epsQ, epsR)) then {
               res = true;
            };
          };
          return res;
};

procedure __computeErrorBoundsAdditionDoit(q, r, p, ops, I, epsTarget, context, switchToAbsolute) {
          var res, P, okay, hopeless, beta;
          var epsAdd, epsR, epsQ;
          var actualEpsAdd, actualEpsQ, actualEpsR, actualEps;
          var actualPolynomial, img;
          var Q, R;
          var etaR, etaQ, alphaQ, alphaR;
          var actualEtaR, actualEtaQ, actualAlphaQ, actualAlphaR;
          var fmt;
          var noCancellationPossible;
	  var II, delta;

          res = { .okay = false, .reason = { .text = "Could not handle addition", .data = { .q = q, .r = r, .p = p, .ops = ops, .I = I, .domain = I, .epsTarget = epsTarget } } };
          okay = false;
          hopeless = false;
          beta = 1;
          while ((!okay) && (!hopeless)) do {
                alphaQ = sup(abs(computeImageRationalFunction(q.p, p, I)));
                alphaR = sup(abs(computeImageRationalFunction(r.p, p, I)));
                epsAdd = inf([(1/ops) * beta * epsTarget]);
                etaQ = alphaQ * (1 + epsAdd);
                etaR = alphaR * (1 + epsAdd);
		if (epsTarget > 0) then {
                   epsR = ((r.ops)/ops) * beta * epsTarget * 1/etaR;
                   epsQ = ((q.ops)/ops) * beta * epsTarget * 1/etaQ;
		} else {
		   epsR = 0;
		   epsQ = 0;
		};
                Q = __computeErrorBoundsRecurse(q, I, epsQ, context, switchToAbsolute);
                if (Q.okay && __hasEps(Q)) then {
                   R = __computeErrorBoundsRecurse(r, I, epsR, context, switchToAbsolute);
                   if (R.okay && __hasEps(R)) then {
                      actualEpsQ = Q.poly.eps;
                      actualEpsR = R.poly.eps;
                      fmt = context.computeOutputFormatAddition(epsAdd, Q.poly.format, R.poly.format);
                      if (fmt == fmt) then {
                          actualEpsAdd = computeErrorBoundsUnifyErrorBound(context.computeBoundAddition(fmt, Q.poly.format, R.poly.format));
                          actualPolynomial = Q.poly.p + R.poly.p;
                          actualAlphaQ = sup(abs(computeImageRationalFunction(Q.poly.p, actualPolynomial, I)));
                          actualAlphaR = sup(abs(computeImageRationalFunction(R.poly.p, actualPolynomial, I)));
			  if (actualEpsQ == 0) then {
			     actualAlphaQ = 1;
			  };
			  if (actualEpsR == 0) then {
			     actualAlphaR = 1;
			  };
                          actualEps = round(actualAlphaQ * actualEpsQ * (1 + actualEpsAdd) +
                                      actualAlphaR * actualEpsR * (1 + actualEpsAdd) +
                                      actualEpsAdd,prec,RU);
                          if ((actualEps == actualEps) && (actualEps <= epsTarget)) then {
			     II = computeImagePolynomial(actualPolynomial, I);
                             img = II * [1 - actualEps; 1 + actualEps];
			     delta = sup(abs(II * actualEps));
                             noCancellationPossible = computeNoCancellationPossible(Q.poly.p, R.poly.p, I, actualEpsQ, actualEpsR);
                             P = { .type = "addition", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .cannotCancel = noCancellationPossible, .epsOp = actualEpsAdd, .epsOpTarget = epsAdd, .delta = delta, .eps = actualEps, .image = img, .domain = I };
                             okay = true;
                          };
                      } else {
                        hopeless = true;
                      };
                   } else {
                      hopeless = true;
                   };
                } else {
                   hopeless = true;
                };
                if ((!okay) && (!hopeless)) then {
                   beta = beta / 1.25;
                };
          };
          if (okay) then {
             res = { .okay = true, .poly = P };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectAdditionDoitIndirect(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
          var res, epsTarget, r;

          res = { .okay = false, .reason = { .text = "Could not handle addition", .data = { .q = q, .r = r, .ops = ops, .p = p, .I = I, .domain = I, .deltaTarget = deltaTarget } } };

	  if (switchToRelative) then {
	     epsTarget = min(1/2, abs(deltaTarget) / sup(abs(I)));
	     if ((epsTarget == epsTarget) && (epsTarget >= 0)) then {
	     	r = __computeErrorBoundsAdditionDoit(q, r, p, ops, I, epsTarget, context, false);
	     	if (r.okay) then {
	     	   res = r;
	     	};
	     };
	  };

	  return res;
};

procedure __computeErrorBoundsAbsoluteDirectAdditionDoitDirect(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
          var res, P, okay, hopeless, beta;
          var epsAdd, deltaR, deltaQ, deltaAdd;
          var actualEpsAdd, actualDeltaAdd, actualDeltaQ, actualDeltaR, actualDelta;
          var actualEpsQ, actualEpsR;
          var actualPolynomial, img;
          var Q, R;
          var fmt;
          var actualEps, noCancellationPossible;

          res = { .okay = false, .reason = { .text = "Could not handle addition", .data = { .q = q, .r = r, .p = p, .ops = ops, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
          okay = false;
          hopeless = false;
          beta = 1;
          while ((!okay) && (!hopeless)) do {
                deltaR = ((r.ops)/ops) * beta * deltaTarget;
                deltaQ = ((q.ops)/ops) * beta * deltaTarget;
                deltaAdd = (1/ops) * beta * deltaTarget;
                epsAdd = round(deltaAdd / sup(abs(computeImagePolynomial(p, I))),prec,RD);
		Q = __computeErrorBoundsAbsoluteDirectRecurse(q, I, deltaQ, context, switchToRelative);
                if (Q.okay) then {
                   R = __computeErrorBoundsAbsoluteDirectRecurse(r, I, deltaR, context, switchToRelative);
                   if (R.okay) then {
		      actualEpsQ = (match (Q.poly) with
		      		   	  { .eps = _eps } : (Q.poly.eps)
				 	  default :         (infty));
		      actualEpsR = (match (R.poly) with
		      		   	  { .eps = _eps } : (R.poly.eps)
				 	  default :         (infty));
                      match (Q.poly) with
                            { .delta = Qdelta } : { actualDeltaQ = Qdelta; }
                            default : {
                                        actualDeltaQ = sup(abs(computeImagePolynomial(Q.poly.p, I))) * actualEpsQ;
                                      };
                      match (R.poly) with
                            { .delta = Rdelta } : { actualDeltaR = Rdelta; }
                            default : {
                                        actualDeltaR = sup(abs(computeImagePolynomial(R.poly.p, I))) * actualEpsR;
                                      };
		      fmt = context.computeOutputFormatAdditionAbsolute(deltaAdd,
									Q.poly.format, R.poly.format);
                      if (fmt == fmt) then {
                          actualPolynomial = Q.poly.p + R.poly.p;
			  actualDeltaAdd = computeErrorBoundsUnifyErrorBound(context.computeBoundAdditionAbsolute(fmt, Q.poly.format, R.poly.format));
                          actualEpsAdd = round(abs(actualDeltaAdd / inf(abs(computeImagePolynomial(actualPolynomial, I)))),prec,RU);
                          actualDelta = round(actualDeltaR + actualDeltaQ + actualDeltaAdd,prec,RU);
                          actualEps = round(actualDelta / inf(abs(computeImagePolynomial(actualPolynomial, I))),prec,RU);
                          if ((actualDelta == actualDelta) && (actualDelta <= deltaTarget)) then {
                             img = computeImagePolynomial(actualPolynomial, I) + [-actualDelta; +actualDelta];
                             noCancellationPossible = computeNoCancellationPossible(Q.poly.p, R.poly.p, I, actualEpsQ, actualEpsR);
			     if ((actualEpsAdd == actualEpsAdd) && (actualEpsAdd >= 0) && (actualEpsAdd != infty) &&
			         (epsAdd == epsAdd) && (epsAdd >= 0) && (epsAdd != infty) &&
				 (actualEps == actualEps) && (actualEps >= 0) && (actualEps != infty)) then {
                             	 P = { .type = "addition", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .cannotCancel = noCancellationPossible, .epsOp = actualEpsAdd, .epsOpTarget = epsAdd, .eps = actualEps, .delta = actualDelta, .deltaOp = actualDeltaAdd, .deltaOpTarget = deltaAdd, .image = img, .domain = I };
		             } else {
                             	 P = { .type = "addition", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .cannotCancel = noCancellationPossible, .delta = actualDelta, .deltaOp = actualDeltaAdd, .deltaOpTarget = deltaAdd, .image = img, .domain = I };
			     };
                             okay = true;
                          };
                      } else {
                        hopeless = true;
                      };
                   } else {
                      hopeless = true;
                   };
                } else {
                   hopeless = true;
                };
                if ((!okay) && (!hopeless)) then {
                   beta = beta / 1.25;
                };
          };
          if (okay) then {
             res = { .okay = true, .poly = P };
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteDirectAdditionDoit(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
          var res, r;

          res = { .okay = false, .reason = { .text = "Could not handle addition", .data = { .q = q, .r = r, .ops = ops, .p = p, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
	  
	  match (context) with
	  	{ .computeOutputFormatAdditionAbsolute = computeOutputFormatAdditionAbsolute,
		  .computeBoundAdditionAbsolute = computeBoundAdditionAbsolute  } : {
												r = __computeErrorBoundsAbsoluteDirectAdditionDoitDirect(q, r, p, ops, I, deltaTarget, context, switchToRelative);
		  					  			       	   }
		default : {
												r = __computeErrorBoundsAbsoluteDirectAdditionDoitIndirect(q, r, p, ops, I, deltaTarget, context, switchToRelative);
			  };

	  if (r.okay) then {
	     res = r;
	  };

          return res;
};

procedure __computeErrorBoundsAddition(q, r, p, ops, I, epsTarget, context, switchToAbsolute) {
	  var res;

	  res = __computeErrorBoundsAdditionDoit(q, r, p, ops, I, epsTarget, context, switchToAbsolute);

	  return res;
};

procedure __computeErrorBoundsAbsoluteDirectAddition(q, r, p, ops, I, deltaTarget, context, switchToRelative) {
	  var res;

	  res = __computeErrorBoundsAbsoluteDirectAdditionDoit(q, r, p, ops, I, deltaTarget, context, switchToRelative);

	  return res;
};

procedure __computeErrorBoundsAbsoluteWithRelativeDoIt(p, polyTree, I, deltaTarget, context, reswitchToAbsolute) {
          var res, J, epsTarget;

          res = { .okay = false };
          J = computeImagePolynomial(p, I);
          epsTarget = round(abs(deltaTarget / (sup(abs(J)))),prec,RD);
          if ((epsTarget == epsTarget) && (!(epsTarget == infty)) && (epsTarget >= 0)) then {
             res = __computeErrorBoundsRecurse(polyTree, I, epsTarget, context, reswitchToAbsolute);
          };

          return res;
};

procedure __computeErrorBoundsAbsoluteWithRelative(polyTree, I, deltaTarget, context, reswitchToAbsolute) {
          var res;

          res = { .okay = false };
          match (polyTree) with
                { .p = p } : {
                                res = __computeErrorBoundsAbsoluteWithRelativeDoIt(p, polyTree, I, deltaTarget, context, reswitchToAbsolute);
                             }
                default :    {
                                res = { .okay = false };
                             };

          return res;
};

procedure __computeErrorBoundsAdditionAbsolute(q, r, p, ops, I, deltaTarget, context)  {
          var res, P, okay, hopeless, beta;
          var epsAdd, deltaR, deltaQ, deltaAdd;
          var actualEpsAdd, actualDeltaAdd, actualDeltaQ, actualDeltaR, actualDelta;
          var actualEpsQ, actualEpsR;
          var actualPolynomial, img;
          var Q, R;
          var fmt;
          var actualEps, noCancellationPossible;

          res = { .okay = false, .reason = { .text = "Could not handle addition", .data = { .q = q, .r = r, .p = p, .ops = ops, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
          okay = false;
          hopeless = false;
          beta = 1;
          while ((!okay) && (!hopeless)) do {
                deltaR = ((r.ops)/ops) * beta * deltaTarget;
                deltaQ = ((q.ops)/ops) * beta * deltaTarget;
                deltaAdd = (1/ops) * beta * deltaTarget;
                epsAdd = round(deltaAdd / sup(abs(computeImagePolynomial(p, I))),prec,RD);
                Q = __computeErrorBoundsAbsoluteRecurse(q, I, deltaQ, context, true);
                if (Q.okay && __hasEps(Q)) then {
                   R = __computeErrorBoundsAbsoluteRecurse(r, I, deltaR, context, true);
                   if (R.okay && __hasEps(R)) then {
                      actualEpsQ = Q.poly.eps;
                      actualEpsR = R.poly.eps;
                      match (Q.poly) with
                            { .delta = Qdelta } : { actualDeltaQ = Qdelta; }
                            default : {
                                        actualDeltaQ = sup(abs(computeImagePolynomial(Q.poly.p, I))) * actualEpsQ;
                                      };
                      match (R.poly) with
                            { .delta = Rdelta } : { actualDeltaR = Rdelta; }
                            default : {
                                        actualDeltaR = sup(abs(computeImagePolynomial(R.poly.p, I))) * actualEpsR;
                                      };
		      if (0 in abs(computeImagePolynomial(p, I))) then {
		      	 match (context) with
			       { .computeOutputFormatAdditionAbsolute = COFAA } : {
											fmt = context.computeOutputFormatAdditionAbsolute(deltaAdd,
																	  Q.poly.format, R.poly.format);
			       	 				      	      	  }
			       default :                                          {
											fmt = context.computeOutputFormatAddition(epsAdd, Q.poly.format, R.poly.format);
										  };
		      } else {
                      	 fmt = context.computeOutputFormatAddition(epsAdd, Q.poly.format, R.poly.format);
		      };
                      if (fmt == fmt) then {
                          actualPolynomial = Q.poly.p + R.poly.p;
			  if (0 in abs(computeImagePolynomial(actualPolynomial, I))) then {
			     match (context) with
			     	   { .computeBoundAdditionAbsolute = CBAA } : {
										actualDeltaAdd = computeErrorBoundsUnifyErrorBound(
											     context.computeBoundAdditionAbsolute(fmt, Q.poly.format, R.poly.format));
									        actualEpsAdd = min(abs(epsAdd), round(abs(actualDeltaAdd / inf(abs(
											     computeImagePolynomial(actualPolynomial, I)))),prec,RU));
				     				     	      }
				   default :                                  {
										actualEpsAdd = computeErrorBoundsUnifyErrorBound(
											     context.computeBoundAddition(fmt, Q.poly.format, R.poly.format));
                             							actualDeltaAdd = sup(abs(computeImagePolynomial(actualPolynomial, I))) * actualEpsAdd;
									      };
			  } else {
                             actualEpsAdd = computeErrorBoundsUnifyErrorBound(context.computeBoundAddition(fmt, Q.poly.format, R.poly.format));
                             actualDeltaAdd = sup(abs(computeImagePolynomial(actualPolynomial, I))) * actualEpsAdd;
			  };
                          actualDelta = round(actualDeltaR + actualDeltaQ + actualDeltaAdd,prec,RU);
                          actualEps = round(actualDelta / inf(abs(computeImagePolynomial(actualPolynomial, I))),prec,RU);
                          if ((actualDelta == actualDelta) && (actualDelta <= deltaTarget)) then {
                             img = computeImagePolynomial(actualPolynomial, I) + [-actualDelta; +actualDelta];
                             noCancellationPossible = computeNoCancellationPossible(Q.poly.p, R.poly.p, I, actualEpsQ, actualEpsR);
                             P = { .type = "addition", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .cannotCancel = noCancellationPossible, .epsOp = actualEpsAdd, .epsOpTarget = epsAdd, .eps = actualEps, .delta = actualDelta, .image = img, .domain = I };
                             okay = true;
                          };
                      } else {
                        hopeless = true;
                      };
                   } else {
                      hopeless = true;
                   };
                } else {
                   hopeless = true;
                };
                if ((!okay) && (!hopeless)) then {
                   beta = beta / 1.25;
                };
          };
          if (okay) then {
             res = { .okay = true, .poly = P };
          };

          return res;
};

procedure __computeErrorBoundsMultiplicationAbsolute(q, r, p, ops, I, deltaTarget, context)  {
          var res, P, okay, hopeless, beta;
          var epsMul, deltaR, deltaQ, deltaMul;
          var actualEpsMul, actualDeltaMul, actualDeltaQ, actualDeltaR, actualDelta;
          var actualEpsQ, actualEpsR;
          var actualPolynomial, img;
          var Q, R;
          var fmt;
          var actualEps;
	  var alphaQ, alphaR, etaR, etaQ;
	  var actualAlphaQ, actualAlphaR;

          res = { .okay = false, .reason = { .text = "Could not handle multiplication", .data = { .q = q, .r = r, .p = p, .ops = ops, .I = I, .domain = I, .deltaTarget = deltaTarget } } };
          okay = false;
          hopeless = false;
          beta = 1;
          while ((!okay) && (!hopeless)) do {
	  	alphaQ = abs(sup(abs(computeImagePolynomial(q.p, I))));
	  	alphaR = abs(sup(abs(computeImagePolynomial(r.p, I))));
	  	etaR = alphaQ;
                deltaR = ((r.ops)/ops) * beta * deltaTarget * 1/etaR;
		etaQ = alphaR + deltaR;
                deltaQ = ((q.ops)/ops) * beta * deltaTarget * 1/etaQ;
                deltaMul = (1/ops) * beta * deltaTarget;
                epsMul = round(deltaMul / sup(abs(computeImagePolynomial(p, I))),prec,RD);
                Q = __computeErrorBoundsAbsoluteRecurse(q, I, deltaQ, context, true);
                if (Q.okay && __hasEps(Q)) then {
                   R = __computeErrorBoundsAbsoluteRecurse(r, I, deltaR, context, true);
                   if (R.okay && __hasEps(R)) then {
                      actualEpsQ = Q.poly.eps;
                      actualEpsR = R.poly.eps;
                      match (Q.poly) with
                            { .delta = Qdelta } : { actualDeltaQ = Qdelta; }
                            default : {
                                        actualDeltaQ = sup(abs(computeImagePolynomial(Q.poly.p, I))) * actualEpsQ;
                                      };
                      match (R.poly) with
                            { .delta = Rdelta } : { actualDeltaR = Rdelta; }
                            default : {
                                        actualDeltaR = sup(abs(computeImagePolynomial(R.poly.p, I))) * actualEpsR;
                                      };
		      if (0 in abs(computeImagePolynomial(p, I))) then {
		      	 match (context) with
			       { .computeOutputFormatMultiplicationAbsolute = COFMA } : {
											fmt = context.computeOutputFormatMultiplicationAbsolute(deltaMul,
																	  Q.poly.format, R.poly.format);
			       	 				      	      	  }
			       default :                                          {
											fmt = context.computeOutputFormatMultiplication(epsMul, Q.poly.format, R.poly.format);
										  };
		      } else {
                      	 fmt = context.computeOutputFormatMultiplication(epsMul, Q.poly.format, R.poly.format);
		      };
                      if (fmt == fmt) then {
                          actualPolynomial = Q.poly.p * R.poly.p;
			  if (0 in abs(computeImagePolynomial(actualPolynomial, I))) then {
			     match (context) with
			     	   { .computeBoundMultiplicationAbsolute = CBMA } : {
										actualDeltaMul = computeErrorBoundsUnifyErrorBound(
											     context.computeBoundMultiplicationAbsolute(fmt, Q.poly.format, R.poly.format));
									        actualEpsMul = min(abs(epsMul), round(abs(actualDeltaMul / inf(abs(
											     computeImagePolynomial(actualPolynomial, I)))),prec,RU));
				     				     	      }
				   default :                                  {
										actualEpsMul = computeErrorBoundsUnifyErrorBound(
											     context.computeBoundMultiplication(fmt, Q.poly.format, R.poly.format));
                             							actualDeltaMul = sup(abs(computeImagePolynomial(actualPolynomial, I))) * actualEpsMul;
									      };
			  } else {
                             actualEpsMul = computeErrorBoundsUnifyErrorBound(context.computeBoundMultiplication(fmt, Q.poly.format, R.poly.format));
                             actualDeltaMul = sup(abs(computeImagePolynomial(actualPolynomial, I))) * actualEpsMul;
			  };
	  	          actualAlphaQ = abs(sup(abs(computeImagePolynomial(Q.poly.p, I))));
	  	          actualAlphaR = abs(sup(abs(computeImagePolynomial(R.poly.p, I))));
                          actualDelta = round(actualAlphaQ * actualDeltaR + actualAlphaR * actualDeltaQ + actualDeltaQ * actualDeltaR + actualDeltaMul,prec,RU);
                          actualEps = round(actualDelta / inf(abs(computeImagePolynomial(actualPolynomial, I))),prec,RU);
                          if ((actualDelta == actualDelta) && (actualDelta <= deltaTarget)) then {
                             img = computeImagePolynomial(actualPolynomial, I) + [-actualDelta; +actualDelta];
                             P = { .type = "multiplication", .q = Q.poly, .r = R.poly, .p = actualPolynomial, .format = fmt, .epsOp = actualEpsMul, .epsOpTarget = epsMul, .eps = actualEps, .delta = actualDelta, .image = img, .domain = I };
                             okay = true;
                          };
                      } else {
                        hopeless = true;
                      };
                   } else {
                      hopeless = true;
                   };
                } else {
                   hopeless = true;
                };
                if ((!okay) && (!hopeless)) then {
                   beta = beta / 1.25;
                };
          };
          if (okay) then {
             res = { .okay = true, .poly = P };
          };

          return res;
};

procedure __synchronizeErrorBounds(errorBoundsTree) {
	  var res, addEps, addDelta, newEps, newDelta, J, poly;

	  res = errorBoundsTree;

	  if (res.okay) then {
	     poly = res.poly;
	     addEps = false;
	     addDelta = false;
	     match (poly) with
	     	   { .eps = eps, .delta = delta } :     {
							        /* Nothing to do */
		     	    	 	  	        }
		   { .p = p, .domain = I, .eps = eps } :     {
							        /* Got eps, no delta */
								J = computeImagePolynomial(p, I);
								newDelta = abs(eps) * sup(abs(J));
								addDelta = true;
						        }
		   { .p = p, .domain = I, .delta = delta } : {
								/* Got delta, no eps */
								J = computeImagePolynomial(p, I);
								if (inf(abs(J)) != 0) then {
								   newEps = abs(abs(delta) / inf(abs(J)));
 								   if (!(newEps == newEps)) then {
								      newEps = infty;
								   };
								} else {
								   newEps = infty;
								};
								addEps = true;
						        }
		   default :                            {
							        /* Cannot do anything */
						        };
	     if (addEps) then {
	     	poly.eps = newEps;
	     };
	     if (addDelta) then {
	     	poly.delta = newDelta;
	     };
	     addEps = false;
	     addDelta = false;
	     match (poly) with
	     	   { .epsOp = eps, .deltaOp = delta } :   {
							        /* Nothing to do */
		     	    	 	  	          }
		   { .p = p, .domain = I, .epsOp = eps } :     {
							        /* Got eps, no delta */
								J = computeImagePolynomial(p, I);
								newDelta = abs(eps) * sup(abs(J));
								addDelta = true;
						          }
		   { .p = p, .domain = I, .deltaOp = delta } : {
		   					        /* Got delta, no eps */
								J = computeImagePolynomial(p, I);
								if (inf(abs(J)) != 0) then {
								   newEps = abs(abs(delta) / inf(abs(J)));
 								   if (!(newEps == newEps)) then {
								      newEps = infty;
								   };
								} else {
								   newEps = infty;
								};
								addEps = true;
						          }
		   default :                              {
							        /* Cannot do anything */
						          };
	     if (addEps) then {
	     	poly.epsOp = newEps;
	     };
	     if (addDelta) then {
	     	poly.deltaOp = newDelta;
	     };
	     res.poly = poly;
	  };

	  return res;
};

procedure __computeErrorBoundsAbsoluteRecurse(polyTree, I, rawDeltaTarget, context, reswitchToAbsolute) {
          var res;
          var deltaTarget;

          deltaTarget = abs(inf(abs([rawDeltaTarget])));
          if (0 in [rawDeltaTarget]) then {
             deltaTarget = 0;
          };

          res = { .okay = false };

          if (deltaTarget >= 0) then {
              match (polyTree) with
                    { .type = "constant", .c = c, .p = p, .ops = ops }               : {
                                                                                          res = __computeErrorBoundsAbsoluteWithRelative(polyTree, I, deltaTarget, context, false);
                                                                                       }
                    { .type = "variable", .p = _x_, .ops = ops }                     : {
                                                                                          res = __computeErrorBoundsAbsoluteWithRelative(polyTree, I, deltaTarget, context, false);
                                                                                       }
                    { .type = "power", .k = k, .p = p, .ops = ops }                  : {
                                                                                          res = __computeErrorBoundsAbsoluteWithRelative(polyTree, I, deltaTarget, context, false);
                                                                                       }
                    { .type = "multiplication", .q = q, .r = r, .p = p, .ops = ops } : {
                                                                                          res = __computeErrorBoundsMultiplicationAbsolute(q, r, p, ops, I, deltaTarget, context);
                                                                                       }
                    { .type = "addition", .q = q, .r = r, .p = p, .ops = ops }       : {
                                                                                          res = __computeErrorBoundsAdditionAbsolute(q, r, p, ops, I, deltaTarget, context);
                                                                                       }
                    default                                                          : {
                                                                                          res = { .okay = false };
                                                                                       };
          };

	  if (res.okay) then {
             res.poly.deltaTarget = deltaTarget;
	  };

          return __synchronizeErrorBounds(res);
};

procedure __computeErrorBoundsAbsolute(polyTree, I, deltaTarget, context) {
	  var res;

	  res = __computeErrorBoundsAbsoluteRecurse(polyTree, I, deltaTarget, context, true);

	  return res;
};

procedure __computeErrorBoundsTrySwitchToAbsoluteDoIt(p, polyTree, I, epsTarget, context) {
          var res, J, deltaTarget;

          res = { .okay = false };
          J = computeImagePolynomial(p, I);
          if (!(0 in J)) then {
             deltaTarget = round(abs(epsTarget * (inf(abs(J)))),prec,RD);
             if ((deltaTarget == deltaTarget) && (!(deltaTarget == infty)) && (deltaTarget > 0)) then {
                res = __computeErrorBoundsAbsolute(polyTree, I, deltaTarget, context);
             };
          };

          return res;
};

procedure __computeErrorBoundsTrySwitchToAbsolute(polyTree, I, epsTarget, context) {
          var res;

          res = { .okay = false };
          match (polyTree) with
                { .p = p } : {
                                res = __computeErrorBoundsTrySwitchToAbsoluteDoIt(p, polyTree, I, epsTarget, context);
                             }
                default :    {
                                res = { .okay = false };
                             };

          return res;
};

procedure __computeErrorBoundsRelative(polyTree, I, epsTarget, context) {
	  var res;

	  res = __computeErrorBoundsRecurse(polyTree, I, epsTarget, context, true);

	  return res;
};

procedure __computeErrorBoundsTrySwitchToRelativeDoIt(p, polyTree, I, deltaTarget, context) {
          var res, J, epsTarget;

          res = { .okay = false };
          J = computeImagePolynomial(p, I);
          epsTarget = round(abs(deltaTarget / (sup(abs(J)))),prec,RD);
          if ((epsTarget == epsTarget) && (!(epsTarget == infty)) && (epsTarget > 0)) then {
             res = __computeErrorBoundsRelative(polyTree, I, epsTarget, context);
          };

          return res;
};

procedure __computeErrorBoundsTrySwitchToRelative(polyTree, I, deltaTarget, context) {
          var res;

          res = { .okay = false };
          match (polyTree) with
                { .p = p } : {
                                res = __computeErrorBoundsTrySwitchToRelativeDoIt(p, polyTree, I, deltaTarget, context);
                             }
                default :    {
                                res = { .okay = false };
                             };

          return res;
};

procedure __computeErrorBoundsRecurse(polyTree, I, rawEpsTarget, context, switchToAbsolute) {
          var res;
          var epsTarget;
          var r;

          epsTarget = abs(inf(abs([rawEpsTarget])));
          if (0 in [rawEpsTarget]) then {
             epsTarget = 0;
          };

          res = { .okay = false };
          if (epsTarget >= 0) then {
              match (polyTree) with
                    { .type = "constant", .c = c, .p = p, .ops = ops }               : {
                                                                                          res = __computeErrorBoundsConstant(c, I, epsTarget, context);
                                                                                       }
                    { .type = "variable", .p = _x_, .ops = ops }                     : {
                                                                                          res = __computeErrorBoundsVariable(I, epsTarget, context);
                                                                                       }
                    { .type = "power", .k = k, .p = p, .ops = ops }                  : {
                                                                                          if (k == 0) then {
                                                                                             res = __computeErrorBoundsConstant(1, I, epsTarget, context);
                                                                                          } else {
                                                                                             if (k == 1) then {
                                                                                                res = __computeErrorBoundsVariable(I, epsTarget, context);
                                                                                             } else {
                                                                                                res = __computeErrorBoundsPower(k, I, epsTarget, context);
                                                                                             };
                                                                                          };
                                                                                       }
                    { .type = "multiplication", .q = q, .r = r, .p = p, .ops = ops } : {
                                                                                          res = __computeErrorBoundsMultiplication(q, r, p, ops, I, epsTarget, context, switchToAbsolute);
                                                                                       }
                    { .type = "addition", .q = q, .r = r, .p = p, .ops = ops }       : {
                                                                                          res = __computeErrorBoundsAddition(q, r, p, ops, I, epsTarget, context, switchToAbsolute);
                                                                                       }
                    default                                                          : {
                                                                                          res = { .okay = false };
                                                                                       };
          };

          if ((!(res.okay)) && switchToAbsolute) then {
             r = __computeErrorBoundsTrySwitchToAbsolute(polyTree, I, epsTarget, context);
             if (r.okay) then {
                res = r;
             };
          };

          if (res.okay) then {
             res.poly.epsTarget = epsTarget;
	     match (res.poly) with
	     	   { .eps = eps } : {
					if (!(res.poly.eps == res.poly.eps)) then {
	     				   res.poly.eps = infty;
	     				};
				    }
	     	   default :        { };
          };

          if (!(res.okay)) then {
             match(res) with
                        { .okay = false, .reason = reason } : { }
                        { .okay = false } : {
                                                res.reason = { .text = "Generic failure", .data = { .polyTree = polyTree, .I = I, .domain = I, .epsTarget = epsTarget } };
                                            }
                        default : { };
          };

          return __synchronizeErrorBounds(res);
};

procedure __computeErrorBoundsAbsoluteDirectRecurse(polyTree, I, rawDeltaTarget, context, switchToRelative) {
          var res;
          var deltaTarget;
          var r;
	  var doWork, J;
	  var altEpsTarget, altEps;

          deltaTarget = abs(inf(abs([rawDeltaTarget])));
          if (0 in [rawDeltaTarget]) then {
             deltaTarget = 0;
          };

	  res = { .okay = false };
          if (deltaTarget >= 0) then {
              match (polyTree) with
                    { .type = "constant", .c = c, .p = p, .ops = ops }               : {
                                                                                          res = __computeErrorBoundsAbsoluteDirectConstant(c, I, deltaTarget, context);
                                                                                       }
                    { .type = "variable", .p = _x_, .ops = ops }                     : {
                                                                                          res = __computeErrorBoundsAbsoluteDirectVariable(I, deltaTarget, context);
                                                                                       }
                    { .type = "power", .k = k, .p = p, .ops = ops }                  : {
                                                                                          if (k == 0) then {
                                                                                             res = __computeErrorBoundsAbsoluteDirectConstant(1, I, deltaTarget, context);
                                                                                          } else {
                                                                                             if (k == 1) then {
                                                                                                res = __computeErrorBoundsAbsoluteDirectVariable(I, deltaTarget, context);
                                                                                             } else {
                                                                                                res = __computeErrorBoundsAbsoluteDirectPower(k, I, deltaTarget, context);
                                                                                             };
                                                                                          };
                                                                                       }
                    { .type = "multiplication", .q = q, .r = r, .p = p, .ops = ops } : {
                                                                                          res = __computeErrorBoundsAbsoluteDirectMultiplication(q, r, p, ops, I, deltaTarget, context, switchToRelative);
                                                                                       }
                    { .type = "addition", .q = q, .r = r, .p = p, .ops = ops }       : {
                                                                                          res = __computeErrorBoundsAbsoluteDirectAddition(q, r, p, ops, I, deltaTarget, context, switchToRelative);
                                                                                       }
                    default                                                          : {
                                                                                          res = { .okay = false };
                                                                                       };
          };

          if ((!(res.okay)) && switchToRelative) then {
             r = __computeErrorBoundsTrySwitchToRelative(polyTree, I, deltaTarget, context);
             if (r.okay) then {
                res = r;
             };
          };

          if (res.okay) then {
             res.poly.deltaTarget = deltaTarget;
	     match (res.poly) with
	     	   { .delta = delta } : {
					if (!(res.poly.delta == res.poly.delta)) then {
	     				   res.poly.delta = infty;
	     				};
				    }
	     	   default :        { };
          };

	  if (res.okay) then {
	     doWork = false;
	     match (res.poly) with
	     	   { .eps = eps } : { }
		   default :        { doWork = true; };
	     match (res.poly) with
	     	   { .epsTarget = epsTarget } : { }
		   default :        { doWork = true; };
             if (doWork) then {
	     	match (res.poly) with
		   { .p = p, .delta = _delta, .deltaTarget = _deltaTarget } : { doWork = true; }
		   default : { doWork = false; };
		if (doWork) then {
		   J = computeImagePolynomial(res.poly.p, I);
		   if (!(0 in J)) then {
		      altEpsTarget = abs(deltaTarget) / sup(abs(J));
		      altEps = abs(res.poly.delta) / inf(abs(J));
		      doWork = false;
		      match (res.poly) with
		      	    { .eps = eps } : { }
			    default :        { doWork = true; };
	              if (doWork && (altEps == altEps)) then {
		      	 res.poly.eps = altEps;
		      };
		      doWork = false;
		      match (res.poly) with
		      	    { .epsTarget = epsTarget } : { }
			    default :        { doWork = true; };
	              if (doWork && (altEpsTarget == altEpsTarget)) then {
		      	 res.poly.epsTarget = altEpsTarget;
		      };
		   };
		};
	     };
	  };

          if (!(res.okay)) then {
             match(res) with
                        { .okay = false, .reason = reason } : { }
                        { .okay = false } : {
                                                res.reason = { .text = "Generic failure", .data = { .polyTree = polyTree, .I = I, .domain = I, .deltaTarget = deltaTarget } };
                                            }
                        default : { };
          };

          return __synchronizeErrorBounds(res);
};

procedure annotateWithOpCount(polyTree) {
          var res, t, Q, R;

          res = { .okay = false };
          match (polyTree) with
                { .type = "constant", .c = c, .p = p }                           : {
                                                                                      res = { .okay = true, .poly = { .type = "constant", .c = c, .p = p, .ops = 1 } };
                                                                                   }
                { .type = "variable", .p = _x_ }                                 : {
                                                                                      res = { .okay = true, .poly = { .type = "variable", .p = _x_, .ops = 1 } };
                                                                                   }
                { .type = "power", .k = k, .p = p }                              : {
                                                                                      if (k >= 0) then {
                                                                                         if (k == 0) then {
                                                                                            t = 1;
                                                                                         } else {
                                                                                            if (k == 1) then {
                                                                                               t = 1;
                                                                                            } else {
                                                                                               t = ceil(log2(k));
                                                                                            };
                                                                                         };
                                                                                         res = { .okay = true, .poly = { .type = "power", .k = k, .p = p, .ops = t } };
                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                { .type = "multiplication", .q = q, .r = r, .p = p }             : {
                                                                                      Q = annotateWithOpCount(q);
                                                                                      if (Q.okay) then {
                                                                                         R = annotateWithOpCount(r);
                                                                                         if (R.okay) then {
                                                                                            res = { .okay = true, .poly = { .type = "multiplication", .q = Q.poly, .r = R.poly, .p = p, .ops = Q.poly.ops + R.poly.ops + 1 } };
                                                                                         } else {
                                                                                            res = { .okay = false };
                                                                                         };
                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                { .type = "addition", .q = q, .r = r, .p = p }                   : {
                                                                                      Q = annotateWithOpCount(q);
                                                                                      if (Q.okay) then {
                                                                                         R = annotateWithOpCount(r);
                                                                                         if (R.okay) then {
                                                                                            res = { .okay = true, .poly = { .type = "addition", .q = Q.poly, .r = R.poly, .p = p, .ops = Q.poly.ops + R.poly.ops + 1 } };
                                                                                         } else {
                                                                                            res = { .okay = false };
                                                                                         };
                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                default                                                          : {
                                                                                      res = { .okay = false };
                                                                                   };

          return res;
};

procedure equalVariants(vA, vB, context) {
          var res;

	  if ((vA.errorBound == relative) && (vB.errorBound == relative)) then {
             res = ((context.compareFormats(vA.format, vB.format) == 0) &&
             	    (context.compareFormats(vA.inputformat, vB.inputformat) == 0) &&
                    (vA.epsOp == vB.epsOp) &&
                    (vA.eps == vB.eps) &&
                    (vA.image == vB.image) &&
                    (vA.domain == vB.domain));
	  } else {
	     if ((vA.errorBound == absolute) && (vB.errorBound == absolute)) then {
             	res = ((context.compareFormats(vA.format, vB.format) == 0) &&
             	       (context.compareFormats(vA.inputformat, vB.inputformat) == 0) &&
                       (vA.deltaOp == vB.deltaOp) &&
                       (vA.delta == vB.delta) &&
                       (vA.image == vB.image) &&
                       (vA.domain == vB.domain));
             } else {
	        res = false;
	     };
	  };

          return res;
};

procedure joinPoweringVariants(pvA, pvB, context) {
          var powVar, i, l, pvi, pwV, pvj, found;

          powVar = pvA;
          l = length(pvB);
          for i from 0 to l-1 do {
              pvi = pvB[i];
              found = false;
              pwV = powVar;
              while ((!found) && (pwV != [||])) do {
                    pvj = head(pwV);
                    pwV = tail(pwV);
                    if (equalVariants(pvi, pvj, context)) then {
                       found = true;
                    };
              };
              if (!found) then {
                 powVar = pvi .: powVar;
              };
          };

          return powVar;
};

procedure insertPoweringVariant(powOrig, pV, context) {
          var res, found, i, l, pwi, pow;

          pow = powOrig;

          res = { .okay = false };

          l = length(pow);
          found = false;
          i = 0;
          while ((!found) && (i < l)) do {
                pwi = pow[i];
                if ((pwi.k == pV.k) && (horner(pwi.p - pV.p) == 0)) then {
                   pow[i] = { .k = pwi.k, .p = pwi.p, .variants = joinPoweringVariants(pwi.variants, pV.variants, context) };
                   res = { .okay = true, .powerings = pow };
                   found = true;
                } else {
                   i = i + 1;
                };
          };
          if (!found) then {
             res = { .okay = true, .powerings = (pV .: pow) };
          };

          return res;
};

procedure joinRequiredPowerings(powA, powB, context) {
          var res, okay, pow, pwB, pB;

          res = { .okay = false };
          okay = true;
          pow = powA;
          pwB = powB;
          while (okay && (pwB != [||])) do {
                pB = head(pwB);
                pwB = tail(pwB);
                R = insertPoweringVariant(pow, pB, context);
                if (R.okay) then {
                  pow = R.powerings;
                } else {
                  okay = false;
                };
          };
          if (okay) then {
             res = { .okay = true, .powerings = pow };
          };

          return res;
};

procedure __computeRequiredPowerings(polyTree, context) {
          var res, t, Q, R, JP;

          res = { .okay = false };
          match (polyTree) with
                { .type = "constant" }                                           : {
                                                                                      res = { .okay = true, .powerings = [||] };
                                                                                   }
                { .type = "variable" }                                           : {
                                                                                      res = { .okay = true, .powerings = [||] };
                                                                                   }
                { .type = "power",
                  .k = k,
                  .p = p,
                  .format = fmt,
                  .inputformat = inputformat,
                  .epsOp = epsOp,
                  .eps = eps,
		  .deltaOp = deltaOp,
		  .delta = delta,
                  .image = image,
                  .domain = dom }                                      : {
                                                                                      if (k >= 0) then {
										      	 if ((epsOp == epsOp) &&
											     (epsOp >= 0) &&
											     (epsOp <= 1/2) &&
											     (eps == eps) &&
											     (eps >= 0) &&
											     (eps <= 1/2)) then {
                                                                                             res = { .okay = true, .powerings = [| { .k = k,
                                                                                                                                     .p = p,
                                                                                                                                     .variants = [| {
																			.format = fmt,
                                                                                                                                                   	.inputformat = inputformat,
																		   	.errorBound = relative,
                                                                                                                                                   	.epsOp = epsOp,
                                                                                                                                                   	.eps = eps,
                                                                                                                                                   	.image = image,
                                                                                                                                                   	.domain = dom
                                                                                                                                                    } |]
                                                                                                                                   }
                                                                                                                                 |]
                                                                                                   };
											 } else {
                                                                                             res = { .okay = true, .powerings = [| { .k = k,
                                                                                                                                     .p = p,
                                                                                                                                     .variants = [| {
																			.format = fmt,
                                                                                                                                                   	.inputformat = inputformat,
																		   	.errorBound = absolute,
                                                                                                                                                   	.deltaOp = deltaOp,
                                                                                                                                                   	.delta = delta,
                                                                                                                                                   	.image = image,
                                                                                                                                                   	.domain = dom
                                                                                                                                                    } |]
                                                                                                                                   }
                                                                                                                                 |]
                                                                                                   };
											 };
                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                { .type = "power",
                  .k = k,
                  .p = p,
                  .format = fmt,
                  .inputformat = inputformat,
                  .epsOp = epsOp,
                  .eps = eps,
                  .image = image,
                  .domain = dom }                                      : {
                                                                                      if (k >= 0) then {
                                                                                         res = { .okay = true, .powerings = [| { .k = k,
                                                                                                                                 .p = p,
                                                                                                                                 .variants = [| {
                                                                                                                                                   .format = fmt,
                                                                                                                                                   .inputformat = inputformat,
																		   .errorBound = relative,
                                                                                                                                                   .epsOp = epsOp,
                                                                                                                                                   .eps = eps,
                                                                                                                                                   .image = image,
                                                                                                                                                   .domain = dom
                                                                                                                                                } |]
                                                                                                                                }
                                                                                                                            |]
                                                                                                };

                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                { .type = "power",
                  .k = k,
                  .p = p,
                  .format = fmt,
                  .inputformat = inputformat,
                  .deltaOp = deltaOp,
                  .delta = delta,
                  .image = image,
                  .domain = dom }                                      : {
                                                                                      if (k >= 0) then {
                                                                                         res = { .okay = true, .powerings = [| { .k = k,
                                                                                                                                 .p = p,
                                                                                                                                 .variants = [| {
                                                                                                                                                   .format = fmt,
                                                                                                                                                   .inputformat = inputformat,
																		   .errorBound = absolute,
                                                                                                                                                   .deltaOp = deltaOp,
                                                                                                                                                   .delta = delta,
                                                                                                                                                   .image = image,
                                                                                                                                                   .domain = dom
                                                                                                                                                } |]
                                                                                                                                }
                                                                                                                            |]
                                                                                                };

                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }										   
                { .type = "multiplication", .q = q, .r = r }                     : {
                                                                                      Q = __computeRequiredPowerings(q, context);
                                                                                      if (Q.okay) then {
                                                                                         R = __computeRequiredPowerings(r, context);
                                                                                         if (R.okay) then {
                                                                                            JP = joinRequiredPowerings(Q.powerings, R.powerings, context);
                                                                                            if (JP.okay) then {
                                                                                               res = { .okay = true, .powerings = JP.powerings };
                                                                                            } else {
                                                                                               res = { .okay = false };
                                                                                            };
                                                                                         } else {
                                                                                            res = { .okay = false };
                                                                                         };
                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                { .type = "addition", .q = q, .r = r }                           : {
                                                                                      Q = __computeRequiredPowerings(q, context);
                                                                                      if (Q.okay) then {
                                                                                         R = __computeRequiredPowerings(r, context);
                                                                                         if (R.okay) then {
                                                                                            JP = joinRequiredPowerings(Q.powerings, R.powerings, context);
                                                                                            if (JP.okay) then {
                                                                                               res = { .okay = true, .powerings = JP.powerings };
                                                                                            } else {
                                                                                               res = { .okay = false };
                                                                                            };
                                                                                         } else {
                                                                                            res = { .okay = false };
                                                                                         };
                                                                                      } else {
                                                                                         res = { .okay = false };
                                                                                      };
                                                                                   }
                default                                                          : {
                                                                                      res = { .okay = false };
                                                                                   };

          return res;
};

procedure annotateWithRequiredPowerPrecalc(polyTree, context) {
          var res;
          var PR;

          PR = __computeRequiredPowerings(polyTree, context);
          if (PR.okay) then {
             res = { .okay = true, .poly = polyTree, .powerings = PR.powerings };
          } else {
             res = { .okay = false };
          };

          return res;
};

procedure computeErrorBoundsDoit(polyTree, I, epsTarget, context) {
          var res, PTOR, EBR;

          res = { .okay = false };

          PTOR = annotateWithOpCount(polyTree);
          if (PTOR.okay) then {
             EBR = __computeErrorBoundsRecurse(PTOR.poly, I, epsTarget, context, true);
             if (EBR.okay) then {
                res = annotateWithRequiredPowerPrecalc(EBR.poly, context);
             } else {
                res = EBR;
             };
          };
	  
          return res;
};

procedure computeErrorBoundsAbsoluteDoit(polyTree, I, deltaTarget, context) {
          var res, PTOR, EBR;

          res = { .okay = false };

          PTOR = annotateWithOpCount(polyTree);
          if (PTOR.okay) then {
             EBR = __computeErrorBoundsAbsoluteDirectRecurse(PTOR.poly, I, deltaTarget, context, true);
             if (EBR.okay) then {
                res = annotateWithRequiredPowerPrecalc(EBR.poly, context);
             } else {
                res = EBR;
             };
          };
	  
          return res;
};

procedure __computeErrorBoundsInternal(polyTree, I, epsTarget, context) {
          var res;

          res = computeErrorBoundsDoit(polyTree, I, epsTarget, context);

          return res;
};

procedure __computeErrorBoundsAbsoluteInternal(polyTree, I, deltaTarget, context) {
          var res;

          res = computeErrorBoundsAbsoluteDoit(polyTree, I, deltaTarget, context);

          return res;
};


/* This function is the one you should be using!!!!

   The input is a structure made out of structures, as
   laid out below.

   The output is a structure with a field .okay indicating
   if everything went fine and a field .poly that is a

   structure containing several fields:

   .type         one of  "constant", "variable", "addition", "multiplication", "power"
   .p            the polynomial implemented
   .eps          a bound on the relative error; caution: the bound might be infinite
   .epsTarget    the target error for this step
   .epsOp        the error of the operation
   .epsOpTarget  the target error of the (one single) operation
   .format       the format the output of the operation is on
   .image        an interval bounding the (evaluated) image
   .domain       an interval bounding the definition domain
   .cannotCancel (for additions only) a boolean indicating whether we are already sure this does not cancel
   .c (for constants only) the constant's value
   .q (for additions and multiplications only) the one operand, also as a structure like this
   .r (for additions and multiplications only) the other operand, also as a structure like this
   .inputformat (for "variable" only) the original format of the variable

   Additional fields

   .delta        with a bound on the absolute error may be provided.
   .deltaTarget  with the target error for this absolute error step

   Attention: the implemented polynomial (in field result.poly.p) may
              differ from the original one. This may change the
              error between the polynomial and the function the polynomial
              is supposed to implement.

   Attention: accesses to _x_ with "variable" may be exact or provoke roundings.
              This is reflected by the .inputformat vs. .format relationship
              and by .epsOp.

   All formats are the opaque formats the functions contained in the
   context are returning.

   Using an opaque type for the formats allows to capture overlap effects
   for double-double and triple-double inside the "format". Renormalizations
   do not need to be added in the tree returned by this function. They
   just need to be inserted by the pretty-printer when an operation's
   error requires renormalization of an operand before using the operand.

*/
procedure computeErrorBounds(polyTree, I, target, mode, context) {
          var res;

          res = { .okay = false };

	  match (mode) with
	  	relative : {
                               res = __computeErrorBoundsInternal(polyTree, I, target, context);
			   }
	        absolute : {
                               res = __computeErrorBoundsAbsoluteInternal(polyTree, I, target, context);
			   }
	        default :  { };

	  return res;
};

procedure __computeCancellations(q, r, p, I) {
          var res, alphaQ, alphaR;

          res = [||];
          alphaQ = sup(abs(computeImageRationalFunction(q, p, I)));
          if (alphaQ == infty) then {
             res = [| { .q = q, .r = r, .p = p, .I = I, .domain = I } |];
          } else {
             alphaR = sup(abs(computeImageRationalFunction(r, p, I)));
             if (alphaR == infty) then {
                res = [| { .q = q, .r = r, .p = p, .I = I, .domain = I } |];
             };
          };
          return res;
};

procedure __containsCancellations(polyTree, I) {
          var res, Q, R, pq, pr, pp;
          var mycancellations;

          res = { .okay = false };
          match (polyTree) with
                    { .type = "constant", .p = p }                       : {
                                                                              res = { .okay = true, .res = false, .p = p, .cancellations = [||] };
                                                                           }
                    { .type = "variable", .p = p }                       : {
                                                                              res = { .okay = true, .res = false, .p = p, .cancellations = [||] };
                                                                           }
                    { .type = "power", .p = p }                          : {
                                                                              res = { .okay = true, .res = false, .p = p, .cancellations = [||] };
                                                                           }
                    { .type = "multiplication", .q = q, .r = r }         : {
                                                                              Q = __containsCancellations(q, I);
                                                                              if (Q.okay) then {
                                                                                 R = __containsCancellations(r, I);
                                                                                 if (R.okay) then {
                                                                                    res = { .okay = true, .res = (Q.res || R.res), .p = horner(Q.p * R.p),
                                                                                            .cancellations = (Q.cancellations @ R.cancellations) };
                                                                                 };
                                                                              };
                                                                           }
                    { .type = "addition", .q = q, .r = r }               : {
                                                                              Q = __containsCancellations(q, I);
                                                                              if (Q.okay) then {
                                                                                 R = __containsCancellations(r, I);
                                                                                 if (R.okay) then {
                                                                                    pq = Q.p;
                                                                                    pr = R.p;
                                                                                    pp = horner(pq + pr);
                                                                                    mycancellations = __computeCancellations(pq, pr, pp, I);
                                                                                    if (mycancellations == [||]) then {
                                                                                        res = { .okay = true, .res = (Q.res || R.res), .p = pp,
                                                                                                .cancellations = (Q.cancellations @ R.cancellations) };
                                                                                    } else {
                                                                                        res = { .okay = true, .res = true, .p = pp,
                                                                                                .cancellations = ((Q.cancellations @ R.cancellations) @ mycancellations) };
                                                                                    };
                                                                                 };
                                                                              };
                                                                           }
                    default                                              : {
                                                                              res = { .okay = false };
                                                                           };

          if (res.okay) then {
	     if (res.res) then {
	     	pp = res.p;
		J = computeImagePolynomial(pp, I);
                if (!(0 in J)) then {
                    v = round(abs(1 / (inf(abs(J)))),prec,RD);
		    if ((v == v) && (v >= 0) && (v != infty)) then {
		       res = { .okay = true, .res = false, .p = pp, .cancellations = [||] };
		    };
	        };
             };
	  };

          return res;
};

procedure containsCancellations(polyTree, I) {
          var r, res;

          res = { .okay = false };

          r = __containsCancellations(polyTree, I);

          if (r.okay) then {
             res = { .okay = true, .res = r.res, .cancellations = r.cancellations };
          };
          return res;
};

procedure hull(I,J) {
          return [ min(inf(I), inf(J)); max(sup(I), sup(J)) ];
};

procedure __generateGappaDomainPoly(polyTree, context) {
          var res, Q, R;

          res = { .okay = false };
          match (polyTree) with
                { .type = "constant", .domain = dom } :                 {  res = { .okay = true, .domain = dom }; }
                { .type = "variable", .domain = dom } :                 {  res = { .okay = true, .domain = dom }; }
                { .type = "power", .domain = dom } :                    {  res = { .okay = true, .domain = dom }; }
                { .type = "addition", .domain = dom, .q = q, .r = r } : {
                                                                           Q = __generateGappaDomainPoly(q, context);
                                                                           if (Q.okay) then {
                                                                              R = __generateGappaDomainPoly(r, context);
                                                                              if (R.okay) then {
                                                                                 res = { .okay = true, .domain = hull(dom,hull(Q.domain, R.domain)) };
                                                                              } else {
                                                                                 res = { .okay = false };
                                                                              };
                                                                           } else {
                                                                              res = { .okay = false };
                                                                           };
                                                                        }
                { .type = "multiplication", .domain = dom, .q = q, .r = r } : {
                                                                           Q = __generateGappaDomainPoly(q, context);
                                                                           if (Q.okay) then {
                                                                              R = __generateGappaDomainPoly(r, context);
                                                                              if (R.okay) then {
                                                                                 res = { .okay = true, .domain = hull(dom,hull(Q.domain, R.domain)) };
                                                                              } else {
                                                                                 res = { .okay = false };
                                                                              };
                                                                           } else {
                                                                              res = { .okay = false };
                                                                           };
                                                                        }
                default :                                               {
                                                                           res = { .okay = false };
                                                                        };
          return res;
};

procedure __generateGappaDomainPowerings(powerings, context) {
          var res, okay, i, l, domain, first, j, ok, ll, f, d;

          res = { .okay = false };
          match (powerings) with
                [||]      : {  res = { .okay = true, .empty = true }; }
                a .: tl   : {
                               okay = true;
                               i = 0;
                               l = length(powerings);
                               first = true;
                               while (okay && (i < l)) do {
                                     match (powerings[i]) with
                                           { .k = k, .variants = variants } : {
                                                                                ll = length(variants);
                                                                                if (ll > 0) then {
                                                                                   ok = true;
                                                                                   j = 0;
                                                                                   f = true;
                                                                                   while (ok && (j < ll)) do {
                                                                                         match (variants[j]) with
                                                                                               { .domain = dom } : {
                                                                                                                        if (f) then {
                                                                                                                           d = dom;
                                                                                                                           f = false;
                                                                                                                        } else {
                                                                                                                           d = hull(d, dom);
                                                                                                                        };
                                                                                                                        j = j + 1;
                                                                                                                   }
                                                                                               default :           { ok = false; };
                                                                                   };
                                                                                   if (ok && (!f)) then {
                                                                                     if (first) then {
                                                                                        domain = d;
                                                                                        first = false;
                                                                                     } else {
                                                                                        domain = hull(domain, d);
                                                                                     };
                                                                                     i = i + 1;
                                                                                   } else {
                                                                                     okay = false;
                                                                                   };
                                                                                } else {
                                                                                   okay = false;
                                                                                };
                                                                              }
                                           default :                          { okay = false; };
                               };
                               if (okay && (!first)) then {
                                  res = { .okay = true, .empty = false, .domain = domain };
                               } else {
                                  res = { .okay = false };
                               };
                            }
                default :   {  res = { .okay = false }; };

          return res;
};

procedure containedInList(e, L, isEqualCheck, context) {
          var ll, i, found;

          found = false;
          ll = L;
          while ((!found) && (ll != [||])) do {
                i = head(ll);
                ll = tail(ll);
                if (isEqualCheck(i,e,context)) then {
                   found = true;
                };
          };

          return found;
};

procedure isEqualRoundedVariable(a, b, context) {
          var res;

          res = false;
          match (a) with
                { .epsOp = epsOpA, .format = formatA, .inputformat = inputformatA } :
                       {
                           match (b) with
                                 { .epsOp = epsOpB, .format = formatB, .inputformat = inputformatB } :
                                        {
                                                res = ((epsOpA == epsOpB) &&
                                                       (context.compareFormats(formatA, formatB) == 0) &&
                                                       (context.compareFormats(inputformatA, inputformatB) == 0));
                                        }
                                default : { res = false; };
                       }
                default : { res = false; };

          return res;
};

procedure combineRoundedVariables(A, B, context) {
          var res, b;

          res = A;
          for b in B do {
              if (!containedInList(b, res, isEqualRoundedVariable, context)) then {
                 res = b .: res;
              };
          };

          return res;
};

procedure __generateGappaImplicationRoundedVariable(polyTree, context) {
          var res, Q, R, rv;

          res = { .okay = false };
          match (polyTree) with
                { .type = "constant" }                        : { res = { .okay = true, .roundedVariables = [||] }; }
                { .type = "power" }                           : { res = { .okay = true, .roundedVariables = [||] }; }
                { .type = "variable", .epsOp = epsOp }        : {
                                                                   if (epsOp == 0) then {
                                                                      res = { .okay = true, .roundedVariables = [||] };
                                                                   } else {
                                                                      rv.type = polyTree.type;
                                                                      rv.epsOp = polyTree.epsOp;
                                                                      rv.p = polyTree.p;
                                                                      rv.eps = polyTree.eps; /* TODO */
                                                                      rv.image = polyTree.image;
                                                                      rv.domain = polyTree.domain;
                                                                      rv.format = polyTree.format;
                                                                      rv.inputformat = polyTree.inputformat;
                                                                      res = { .okay = true, .roundedVariables = [| rv |] };
                                                                   };
                                                                }
                { .type = "addition", .q = q, .r = r }        : {
                                                                   Q = __generateGappaImplicationRoundedVariable(q, context);
                                                                   if (Q.okay) then {
                                                                      R = __generateGappaImplicationRoundedVariable(r, context);
                                                                      if (R.okay) then {
                                                                         res = { .okay = true,
                                                                                 .roundedVariables = combineRoundedVariables(Q.roundedVariables,
                                                                                                                             R.roundedVariables, context) };
                                                                      } else {
                                                                         res = { .okay = false };
                                                                      };
                                                                   } else {
                                                                      res = { .okay = false };
                                                                   };
                                                                }
                { .type = "multiplication", .q = q, .r = r }  : {
                                                                   Q = __generateGappaImplicationRoundedVariable(q, context);
                                                                   if (Q.okay) then {
                                                                      R = __generateGappaImplicationRoundedVariable(r, context);
                                                                      if (R.okay) then {
                                                                         res = { .okay = true,
                                                                                 .roundedVariables = combineRoundedVariables(Q.roundedVariables,
                                                                                                                             R.roundedVariables, context) };
                                                                      } else {
                                                                         res = { .okay = false };
                                                                      };
                                                                   } else {
                                                                      res = { .okay = false };
                                                                   };
                                                                }
                default                                       : { res = { .okay = false }; };

          return res;
};

procedure gappaInterval(I) {
          var oldDisplay, res;

          oldDisplay = display;
          display = dyadic!;
          res = "[" @ inf(I) @ "," @ sup(I) @ "]";
          display = oldDisplay!;

          return res;
};

procedure gappaIntervalFromEps(eps) {
          return gappaInterval([-eps;eps]);
};

procedure di(i) {
          var res, oldDisplay, oldPrec;

          if (i < 0) then {
             res = "minus" @ di(-i);
          } else {
             if ((floor(i) == i) && (i < 2^128)) then {
                oldDisplay = display;
                display = decimal!;
                res = "" @ i;
                display = oldDisplay!;
             } else {
                oldDisplay = display;
                display = dyadic!;
                oldPrec = prec;
                prec = max(2000, precision(i))!;
                res = "" @ i;
                display = oldDisplay!;
                prec = oldPrec!;
             };
          };
          return res;
};

procedure gappaPowerX(k) {
          var res, i;

          if (k == 0) then {
             res = "1";
          } else {
             if (k == 1) then {
                res = "x";
             } else {
                res = "x";
                for i from 2 to k do {
                    res = res @ " * x";
                };
             };
          };
          return res;
};

procedure __generateGappaImplication(prefix, polyTree, powerings, context) {
          var res, gappaDomainPoly, gappaDomainPowerings, domain;
          var gappaImplicationRoundedVariable;
          var ssaPart, implicationPartHypotheses, i, rv, pwr, rvs, vrts, k, vrt;
          var hints;
          var eps;

          res = { .okay = false };
          gappaDomainPoly = __generateGappaDomainPoly(polyTree, context);
          if (gappaDomainPoly.okay) then {
             gappaDomainPowerings = __generateGappaDomainPowerings(powerings, context);
             if (gappaDomainPowerings.okay) then {
                if (gappaDomainPowerings.empty) then {
                   domain = gappaDomainPoly.domain;
                } else {
                   domain = hull(gappaDomainPoly.domain, gappaDomainPowerings.domain);
                };
                match (polyTree) with
                      { .eps = epsBound, .epsTarget = epsTarget } : { /* TODO */
                                           eps = epsTarget;
                                           if (abs(epsBound) < 1/2 * abs(epsTarget)) then {
                                              eps = 1/2 * abs(epsTarget);
                                           };
                                           eps = round(eps,24,RU);
                                           gappaImplicationRoundedVariable = __generateGappaImplicationRoundedVariable(polyTree, context);
                                           if (gappaImplicationRoundedVariable.okay) then {
                                              ssaPart = "";
                                              hints = "";
                                              implicationPartHypotheses = "x in " @ gappaInterval(domain) @ "\n";
                                              if ((0 in domain) && (domain != [0])) then {
                                                  if (inf(domain) == 0) then {
                                                     exclude = [0;1b-1075];
                                                  } else {
                                                     if (sup(domain) == 0) then {
                                                        exclude = [-1b-1075;0];
                                                     } else {
                                                        exclude = [-1b-1075;1b-1075];
                                                     };
                                                  };
                                                  implicationPartHypotheses = implicationPartHypotheses @ "/\\ not x in " @ gappaInterval(exclude) @ "\n";
                                              };
                                              implicationPartEpsilon = "epsilon in " @ gappaIntervalFromEps(eps);
                                              rvs = gappaImplicationRoundedVariable.roundedVariables;
                                              for i from 0 to length(rvs) - 1 do {
                                                  rv = rvs[i];
                                                  ssaPart = ssaPart @ (
                                                              prefix @ "epsilon_rounded_x_" @ di(i) @ " = (" @ prefix @ "rounded_x_" @ di(i) @ " - x)/x;\n"
                                                            );
                                                  implicationPartHypotheses = implicationPartHypotheses @ (
                                                              "/\\ " @ prefix @ "epsilon_rounded_x_" @ di(i) @ " in " @ gappaIntervalFromEps(rv.epsOp) @ "\n"
                                                            );
                                                  hints = hints @ (
                                                              prefix @ "rounded_x_" @ di(i) @ " ~ x;\n" @
                                                              prefix @ "rounded_x_" @ di(i) @ " -> x + x * " @ prefix @ "epsilon_rounded_x_" @ di(i) @ ";\n"
                                                            );
                                              };
                                              for pwr in powerings do {
                                                  k = pwr.k;
                                                  ssaPart = ssaPart @ (
                                                                prefix @ "Mpow_x_" @ di(k) @ " = " @ gappaPowerX(k) @ ";\n"
                                                            );
                                                  vrts = pwr.variants;
                                                  for i from 0 to length(vrts) - 1 do {
                                                      vrt = vrts[i];
                                                      ssaPart = ssaPart @ (
                                                                prefix @ "epsilon_pow_x_" @ di(k) @ "_" @ di(i) @
                                                                " = (" @ prefix @ "pow_x_" @ di(k) @ "_" @ di(i) @ " - " @ prefix @ "Mpow_x_" @ di(k) @ ") / " @ prefix @ "Mpow_x_" @ di(k) @ ";\n"
                                                            );
                                                      implicationPartHypotheses = implicationPartHypotheses @ (
                                                                "/\\ " @ prefix @ "epsilon_pow_x_" @ di(k) @ "_" @ di(i) @ " in " @ gappaIntervalFromEps(vrt.epsOp) @ "\n"
                                                            );
                                                      hints = hints @ (
                                                                prefix @ "pow_x_" @ di(k) @ "_" @ di(i) @ " ~ " @ prefix @ "Mpow_x_" @ di(k) @ ";\n" @
                                                                prefix @ "pow_x_" @ di(k) @ "_" @ di(i) @ " -> " @ prefix @ "Mpow_x_" @ di(k) @
                                                                " + " @ prefix @ "epsilon_pow_x_" @ di(k) @ "_" @ di(i) @ " * " @ prefix @ "Mpow_x_" @ di(k) @ ";\n"
                                                            );
                                                  };
                                              };
                                              res = { .okay = true,
                                                      .ssaPart = ssaPart,
                                                      .implicationPartHypotheses = implicationPartHypotheses,
                                                      .implicationPartEpsilon = implicationPartEpsilon,
                                                      .hints = hints,
                                                      .roundedVariables = gappaImplicationRoundedVariable.roundedVariables,
                                                      .domain = domain };
                                           };
                                       }
                      default        : {
                                           res = { .okay = false };
                                       };
             };
          };

          return res;
};

procedure gappaConstant(c) {
          var res, oldDisplay, oldPrec, p;

          oldDisplay = display;
          oldPrec = prec;
          display = dyadic!;
          prec = max(4000, precision(c))!;

          res = "" @ c;

          prec = oldPrec!;
          display = oldDisplay!;

          return res;
};

procedure __generateGappaConstant(prefix, c, context, ssa, hints, ssano, poly) {
          var res, mySsa, myHints;
          var tmp;

          mySsa = ssa;
          mySsa = mySsa @ prefix @ "tmp_" @ di(ssano) @ " = " @ gappaConstant(c) @ ";\n";
          mySsa = mySsa @ prefix @ "Mtmp_" @ di(ssano) @ " = " @ gappaConstant(c) @ ";\n";
          mySsa = mySsa @ prefix @ "epsilon_" @ di(ssano) @ " = 0;\n";

          myHints = hints;
          myHints = myHints @ prefix @ "tmp_" @ di(ssano) @ " ~ " @ prefix @ "Mtmp_" @ di(ssano) @ ";\n";

          tmp = poly;
          tmp.name = prefix @ "tmp_" @ di(ssano);

          res = { .okay = true, .ssa = mySsa, .ssano = ssano, .hints = myHints, .nextssano = ssano + 1, .poly = tmp };

          return res;
};

procedure __generateGappaVariable(prefix, epsOp, format, inputformat, context, ssa, hints, ssano, roundedVariables, poly) {
          var res, mySsa, myHints, found, i, ll, rv;
          var tmp;

          if (epsOp >= 0) then {
             if (epsOp == 0) then {
                mySsa = ssa;
                mySsa = mySsa @ prefix @ "tmp_" @ di(ssano) @ " = x;\n";
                mySsa = mySsa @ prefix @ "Mtmp_" @ di(ssano) @ " = x;\n";
                mySsa = mySsa @ prefix @ "epsilon_" @ di(ssano) @ " = 0;\n";

                myHints = hints;
                myHints = myHints @ prefix @ "tmp_" @ di(ssano) @ " ~ " @ prefix @ "Mtmp_" @ di(ssano) @ ";\n";

                tmp = poly;
                tmp.name = prefix @ "tmp_" @ di(ssano);

                res = { .okay = true, .ssa = mySsa, .ssano = ssano, .hints = myHints, .nextssano = ssano + 1, .poly = tmp };
             } else {
                found = false;
                i = 0;
                ll = length(roundedVariables);
                while ((!found) && (i < ll)) do {
                      rv = roundedVariables[i];
                      if ((rv.epsOp == epsOp) &&
                          (context.compareFormats(rv.format, format) == 0) &&
                          (context.compareFormats(rv.inputformat, inputformat) == 0)) then {
                          found = true;
                      } else {
                          i = i + 1;
                      };
                };
                if (found) then {
                   mySsa = ssa;
                   mySsa = mySsa @ prefix @ "tmp_" @ di(ssano) @ " = " @ prefix @ "rounded_x_" @ di(i) @ ";\n";
                   mySsa = mySsa @ prefix @ "Mtmp_" @ di(ssano) @ " = x;\n";
                   mySsa = mySsa @ prefix @ "epsilon_" @ di(ssano) @ " = " @ prefix @ "epsilon_rounded_x_" @ di(i) @ ";\n";

                   myHints = hints;
                   myHints = myHints @ prefix @ "tmp_" @ di(ssano) @ " ~ " @ prefix @ "Mtmp_" @ di(ssano) @ ";\n";

                   tmp = poly;
                   tmp.name = prefix @ "tmp_" @ di(ssano);

                   res = { .okay = true, .ssa = mySsa, .ssano = ssano, .hints = myHints, .nextssano = ssano + 1, .poly = tmp };
                } else {
                   res = { .okay = false };
                };
             };
          } else {
             res = { .okay = false };
          };

          return res;
};

procedure __generateGappaPower(prefix, k, format, inputformat, powerings, context, ssa, hints, ssano, poly) {
          var res;
          var fk, fv, i, j, vrts, vrt, pwr, lpwr, lvrts;
          var mySsa, myHints;
          var tmp;

          res = { .okay = false };
          fk = false;
          i = 0;
          lpwr = length(powerings);
          while ((!fk) && (i < lpwr)) do {
                pwr = powerings[i];
                if (pwr.k == k) then {
                   fk = true;
                } else {
                   i = i + 1;
                };
          };
          if (fk) then {
             fv = false;
             j = 0;
             vrts = pwr.variants;
             lvrts = length(vrts);
             while ((!fv) && (j < lvrts)) do {
                   vrt = vrts[j];
                   if ((context.compareFormats(vrt.format, format) == 0) &&
                       (context.compareFormats(vrt.format, format) == 0)) then {
                      fv = true;
                   } else {
                      j = j + 1;
                   };
             };
             if (fv) then {
                mySsa = ssa;
                mySsa = mySsa @ prefix @ "tmp_" @ di(ssano) @ " = " @ prefix @ "pow_x_" @ di(k) @ "_" @ di(j) @ ";\n";
                mySsa = mySsa @ prefix @ "Mtmp_" @ di(ssano) @ " = " @ prefix @ "Mpow_x_" @ di(k) @ ";\n";
                mySsa = mySsa @ prefix @ "epsilon_" @ di(ssano) @ " = " @ prefix @ "epsilon_pow_x_" @ di(k) @ "_" @ di(j) @ ";\n";

                myHints = hints;
                myHints = myHints @ prefix @ "tmp_" @ di(ssano) @ " ~ " @ prefix @ "Mtmp_" @ di(ssano) @ ";\n";

                tmp = poly;
                tmp.name = prefix @ "tmp_" @ di(ssano);

                res = { .okay = true, .ssa = mySsa, .ssano = ssano, .hints = myHints, .nextssano = ssano + 1, .poly = tmp };
             };
          };

          return res;
};

procedure __generateGappaAddition(prefix, epsOp, q, r, powerings, context, ssa, hints, ssano, roundedVariables, dom, poly) {
          var res, Q, R, myssano, myssa, myhints, qssano, rssano;
          var bitAccuracy;
          var pq, pr, pqokay, prokay;
          var tmp;

          res = { .okay = false };
          pqokay = false;
          prokay = false;
          match (q) with
                { .p = p } : { pq = p; pqokay = true; }
                default    : { pqokay = false; };
          match (r) with
                { .p = p } : { pr = p; prokay = true; }
                default    : { prokay = false; };
          if (pqokay && prokay) then {
             if (epsOp >= 0) then {
                Q = __generateGappa(prefix, q, powerings, context, ssa, hints, ssano, roundedVariables, dom);
                if (Q.okay) then {
                   R = __generateGappa(prefix, r, powerings, context, Q.ssa, Q.hints, Q.nextssano, roundedVariables, dom);
                   if (R.okay) then {
                      myssano = R.nextssano;
                      myssa = R.ssa;
                      myhints = R.hints;
                      qssano = Q.ssano;
                      rssano = R.ssano;

                      if (epsOp == 0) then {
                         myssa = myssa @ prefix @ "tmp_" @ di(myssano) @ " = " @ prefix @ "tmp_" @ di(Q.ssano) @ " + " @ prefix @ "tmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "Mtmp_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " + " @ prefix @ "Mtmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_" @ di(myssano) @ " = (" @ prefix @ "tmp_" @ di(myssano) @ " - " @ prefix @ "Mtmp_" @ di(myssano) @ ") / " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_prop_" @ di(myssano) @ " = ((" @ prefix @ "tmp_" @ di(Q.ssano) @ " + " @ prefix @ "tmp_" @ di(R.ssano) @ ") - " @ prefix @ "Mtmp_" @ di(myssano) @
                                   ") / " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myssa = myssa @ prefix @ "alpha_q_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " / (" @ prefix @ "Mtmp_" @ di(Q.ssano) @ " + " @ prefix @ "Mtmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "alpha_r_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(R.ssano) @ " / (" @ prefix @ "Mtmp_" @ di(Q.ssano) @ " + " @ prefix @ "Mtmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "delta_q_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " / " @ prefix @ "Mtmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "delta_r_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(R.ssano) @ " / " @ prefix @ "Mtmp_" @ di(Q.ssano) @ ";\n";

                         myhints = myhints @ prefix @ "tmp_" @ di(myssano) @ " ~ " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myhints = myhints @ prefix @ "epsilon_" @ di(myssano) @ " -> " @ prefix @ "epsilon_prop_" @ di(myssano);
                         myhints = myhints @ prefix @ "epsilon_prop_" @ di(myssano) @ " -> " @ prefix @ "alpha_q_" @ di(myssano) @ " * " @ prefix @ "epsilon_" @ di(Q.ssano) @
                                   " + " @ prefix @ "alpha_r_" @ di(myssano) @ " * " @ prefix @ "epsilon_" @ di(R.ssano) @ ";\n";
                         myhints = myhints @ prefix @ "alpha_q_" @ di(myssano) @ " -> " @ prefix @ "delta_q_" @ di(myssano) @ " * (1 / (1 + " @ prefix @ "delta_q_" @ di(myssano) @ "));\n";
                         myhints = myhints @ prefix @ "alpha_r_" @ di(myssano) @ " -> " @ prefix @ "delta_r_" @ di(myssano) @ " * (1 / (1 + " @ prefix @ "delta_r_" @ di(myssano) @ "));\n";
                      } else {
                         bitAccuracy = floor(-log2(epsOp));

                         myssa = myssa @ prefix @ "tmp_" @ di(myssano) @ " = add_rel<" @ di(bitAccuracy) @ ">(" @ prefix @ "tmp_" @ di(Q.ssano) @ ", " @ prefix @ "tmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "Mtmp_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " + " @ prefix @ "Mtmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_" @ di(myssano) @ " = (" @ prefix @ "tmp_" @ di(myssano) @ " - " @ prefix @ "Mtmp_" @ di(myssano) @ ") / " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_add_" @ di(myssano) @ " = (" @ prefix @ "tmp_" @ di(myssano) @ " - (" @ prefix @ "tmp_" @ di(Q.ssano) @
                                   " + " @ prefix @ "tmp_" @ di(R.ssano) @ ")) / (" @ prefix @ "tmp_" @ di(Q.ssano) @ " + " @ prefix @ "tmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "epsilon_prop_" @ di(myssano) @ " = ((" @ prefix @ "tmp_" @ di(Q.ssano) @ " + " @ prefix @ "tmp_" @ di(R.ssano) @ ") - " @ prefix @ "Mtmp_" @ di(myssano) @
                                   ") / " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myssa = myssa @ prefix @ "alpha_q_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " / (" @ prefix @ "Mtmp_" @ di(Q.ssano) @ " + " @ prefix @ "Mtmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "alpha_r_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(R.ssano) @ " / (" @ prefix @ "Mtmp_" @ di(Q.ssano) @ " + " @ prefix @ "Mtmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "delta_q_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " / " @ prefix @ "Mtmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "delta_r_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(R.ssano) @ " / " @ prefix @ "Mtmp_" @ di(Q.ssano) @ ";\n";

                         myhints = myhints @ prefix @ "tmp_" @ di(myssano) @ " ~ " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myhints = myhints @ prefix @ "epsilon_" @ di(myssano) @ " -> " @ prefix @ "epsilon_add_" @ di(myssano) @ " + " @ prefix @ "epsilon_prop_" @ di(myssano) @ " + " @ prefix @ "epsilon_add_" @ di(myssano) @
                                   " * " @ prefix @ "epsilon_prop_" @ di(myssano) @ ";\n";
                         myhints = myhints @ prefix @ "epsilon_prop_" @ di(myssano) @ " -> " @ prefix @ "alpha_q_" @ di(myssano) @ " * " @ prefix @ "epsilon_" @ di(Q.ssano) @
                                   " + " @ prefix @ "alpha_r_" @ di(myssano) @ " * " @ prefix @ "epsilon_" @ di(R.ssano) @ ";\n";
                         myhints = myhints @ prefix @ "alpha_q_" @ di(myssano) @ " -> " @ prefix @ "delta_q_" @ di(myssano) @ " * (1 / (1 + " @ prefix @ "delta_q_" @ di(myssano) @ "));\n";
                         myhints = myhints @ prefix @ "alpha_r_" @ di(myssano) @ " -> " @ prefix @ "delta_r_" @ di(myssano) @ " * (1 / (1 + " @ prefix @ "delta_r_" @ di(myssano) @ "));\n";
                      };

                      tmp = poly;
                      tmp.q = Q.poly;
                      tmp.r = R.poly;
                      tmp.name = prefix @ "tmp_" @ di(myssano);

                      res = { .okay = true, .ssa = myssa, .ssano = myssano, .hints = myhints, .nextssano = myssano + 1, .poly = tmp };
                   };
                };
             };
          };

          return res;
};

procedure __generateGappaMultiplication(prefix, epsOp, q, r, powerings, context, ssa, hints, ssano, roundedVariables, dom, poly) {
          var res, Q, R, myssano, myssa, myhints, qssano, rssano;
          var bitAccuracy;
          var pq, pr, pqokay, prokay;
          var tmp;

          res = { .okay = false };
          pqokay = false;
          prokay = false;
          match (q) with
                { .p = p } : { pq = p; pqokay = true; }
                default    : { pqokay = false; };
          match (r) with
                { .p = p } : { pr = p; prokay = true; }
                default    : { prokay = false; };
          if (pqokay && prokay) then {
             if (epsOp >= 0) then {
                Q = __generateGappa(prefix, q, powerings, context, ssa, hints, ssano, roundedVariables, dom);
                if (Q.okay) then {
                   R = __generateGappa(prefix, r, powerings, context, Q.ssa, Q.hints, Q.nextssano, roundedVariables, dom);
                   if (R.okay) then {
                      myssano = R.nextssano;
                      myssa = R.ssa;
                      myhints = R.hints;
                      qssano = Q.ssano;
                      rssano = R.ssano;

                      if (epsOp == 0) then {
                         myssa = myssa @ prefix @ "tmp_" @ di(myssano) @ " = " @ prefix @ "tmp_" @ di(Q.ssano) @ " * " @ prefix @ "tmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "Mtmp_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " * " @ prefix @ "Mtmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_" @ di(myssano) @ " = (" @ prefix @ "tmp_" @ di(myssano) @ " - " @ prefix @ "Mtmp_" @ di(myssano) @ ") / " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";

                         myhints = myhints @ prefix @ "tmp_" @ di(myssano) @ " ~ " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myhints = myhints @ prefix @ "epsilon_" @ di(myssano) @ " -> " @ prefix @ "epsilon_" @ di(Q.ssano) @
                                                          " + " @ prefix @ "epsilon_" @ di(R.ssano) @ " + " @ prefix @ "epsilon_" @ di(Q.ssano) @ " * " @ prefix @ "epsilon_" @ di(R.ssano) @ ";\n";

                      } else {
                         bitAccuracy = floor(-log2(epsOp));

                         myssa = myssa @ prefix @ "tmp_" @ di(myssano) @ " = mul_rel<" @ di(bitAccuracy) @ ">(" @ prefix @ "tmp_" @ di(Q.ssano) @ ", " @ prefix @ "tmp_" @ di(R.ssano) @ ");\n";
                         myssa = myssa @ prefix @ "Mtmp_" @ di(myssano) @ " = " @ prefix @ "Mtmp_" @ di(Q.ssano) @ " * " @ prefix @ "Mtmp_" @ di(R.ssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_" @ di(myssano) @ " = (" @ prefix @ "tmp_" @ di(myssano) @ " - " @ prefix @ "Mtmp_" @ di(myssano) @ ") / " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myssa = myssa @ prefix @ "epsilon_mul_" @ di(myssano) @ " = (" @ prefix @ "tmp_" @ di(myssano) @ " - " @ prefix @ "tmp_" @ di(Q.ssano) @
                                   " * " @ prefix @ "tmp_" @ di(R.ssano) @ ") / (" @ prefix @ "tmp_" @ di(Q.ssano) @ " * " @ prefix @ "tmp_" @ di(R.ssano) @ ");\n";

                         myhints = myhints @ prefix @ "tmp_" @ di(myssano) @ " ~ " @ prefix @ "Mtmp_" @ di(myssano) @ ";\n";
                         myhints = myhints @ prefix @ "epsilon_" @ di(myssano) @ " -> " @ prefix @ "epsilon_" @ di(Q.ssano) @ " + " @ prefix @ "epsilon_" @ di(R.ssano) @ " + " @ prefix @ "epsilon_mul_" @ di(myssano) @
                                   " + " @ prefix @ "epsilon_" @ di(Q.ssano) @ " * " @ prefix @ "epsilon_" @ di(R.ssano) @ " + " @ prefix @ "epsilon_" @ di(Q.ssano) @ " * " @ prefix @ "epsilon_mul_" @ di(myssano) @
                                   " + " @ prefix @ "epsilon_" @ di(R.ssano) @ " * " @ prefix @ "epsilon_mul_" @ di(myssano) @ " + " @ prefix @ "epsilon_" @ di(Q.ssano) @ " * " @ prefix @ "epsilon_" @ di(R.ssano) @
                                   " * " @ prefix @ "epsilon_mul_" @ di(myssano) @ ";\n";
                      };

                      tmp = poly;
                      tmp.q = Q.poly;
                      tmp.r = R.poly;
                      tmp.name = prefix @ "tmp_" @ di(myssano);

                      res = { .okay = true, .ssa = myssa, .ssano = myssano, .hints = myhints, .nextssano = myssano + 1, .poly = tmp };
                   };
                };
             };
          };

          return res;
};

procedure __generateGappa(prefix, polyTree, powerings, context, ssa, hints, ssano, roundedVariables, dom) {
          var res, Q, R;

          res = { .okay = false };
          match (polyTree) with
                { .type = "constant", .c = c }                                            : {
                                                                                              res = __generateGappaConstant(prefix, c, context, ssa, hints, ssano, polyTree);
                                                                                            }
                { .type = "variable", .epsOp = epsOp, .format = format, .inputformat = inputformat } : {
                                                                                              res = __generateGappaVariable(prefix, epsOp, format, inputformat, context, ssa, hints, ssano, roundedVariables, polyTree);
                                                                                            }
                { .type = "power", .k = k, .format = format, .inputformat = inputformat } : {
                                                                                              res = __generateGappaPower(prefix, k, format, inputformat, powerings, context, ssa, hints, ssano, polyTree);
                                                                                            }
                { .type = "addition", .epsOp = epsOp, .q = q, .r = r }                    : {
                                                                                              res = __generateGappaAddition(prefix, epsOp, q, r, powerings, context, ssa, hints, ssano, roundedVariables, dom, polyTree);
                                                                                            }
                { .type = "multiplication", .epsOp = epsOp, .q = q, .r = r }              : {
                                                                                              res = __generateGappaMultiplication(prefix, epsOp, q, r, powerings, context, ssa, hints, ssano, roundedVariables, dom, polyTree);
                                                                                            }
                default                                                                   : { res = { .okay = false }; };

          return res;
};

procedure __generateGappaRewritePowerings(prefix, powerings) {
          var res, i, lp, pwr, npwr, vrts, vrt, lvrts, nvrts, nvrt, j, k;

          lp = length(powerings);
          res = [||];
          for i from 0 to lp - 1 do {
              pwr = powerings[i];
              vrts = pwr.variants;
              lvrts = length(vrts);
              nvrts = [||];
              k = pwr.k;
              for j from 0 to lvrts - 1 do {
                  vrt = vrts[j];
                  nvrt = vrt;
                  nvrt.name = prefix @ "pow_x_" @ di(k) @ "_" @ di(j);
                  nvrts = nvrt .: nvrts;
              };
              nvrts = revert(nvrts);
              npwr = pwr;
              npwr.variants = nvrts;
              res = npwr .: res;
          };
          res = revert(res);
          return res;
};

procedure __generateGappaGetSplitpointsDoit(a, b) {
	  var res, r, i, la, lb;

	  if (b < a) then {
	     res = [||];
	  } else {
             if (b - a < 1b-1075) then {
	     	res = [| round((a + b)/2, prec, RN) |];
	     } else {
	        if (a == 0) then {
		   res = __generateGappaGetSplitpointsDoit(0,0)  @ __generateGappaGetSplitpointsDoit(1b-1075,b);
		} else {
		   if (b == 0) then {
		      res = __generateGappaGetSplitpointsDoit(a,-1b-1075)  @ __generateGappaGetSplitpointsDoit(0,0);
		   } else {
		      if (a * b <= 0) then {
		         res = __generateGappaGetSplitpointsDoit(a,0)  @ __generateGappaGetSplitpointsDoit(0,b);
		      } else {
		      	 if (b < 0) then {
			    r = __generateGappaGetSplitpointsDoit(-b,-a);
			    res = [||];
			    for i in r do {
			    	res = (-i) .: res;
			    };
			 } else {
                            la = floor(log2(abs(a)));
			    lb = ceil(log2(abs(b)));
			    res = [||];
			    for i from la to lb by ceil((lb - la) / 15) do {
			    	r = 2^i;
				if ((a <= r) && (r <= b)) then {
				   res = r .: res;
				};
			    };
			 };
		      };
		   };
		};
	     };
	  };
	  return res;
};



procedure __generateGappaGetSplitpoints(dom) {
     return sort(__generateGappaGetSplitpointsDoit(inf(dom), sup(dom)));
};

/* Call this function to get some Gappa proof */
procedure generateGappa(prefix, polyTree, powerings, context) {
          var res, gappaImplication, gappa;
          var dom, splitpoints, s, first, splitpointlist;

          res = { .okay = false, .poly = polyTree, .powerings = powerings };
          gappaImplication = __generateGappaImplication(prefix, polyTree, powerings, context);
          if (gappaImplication.okay) then {
             gappa = __generateGappa(prefix, polyTree, powerings, context, "", "", 1, gappaImplication.roundedVariables, gappaImplication.domain);
             if (gappa.okay) then {
                dom = gappaImplication.domain;
                first = true;
                splitpoints = "";
		splitpointlist = __generateGappaGetSplitpoints(dom);
		if (splitpointlist == [||]) then {
		   splitpointlist = [| round(mid(dom),prec,RN) |];
		};
                for s in splitpointlist do {
                        if (first) then {
                           splitpoints = gappaConstant(round(s,24,RN));
                           first = false;
                        } else {
                           splitpoints = splitpoints @ ", " @ gappaConstant(round(s,24,RN));
                        };
                };
                res = { .okay = true,
                        .gappa = (
                                   "#@-Wno-dichotomy-failure\n" @
                                   "#@-Wno-null-denominator\n" @
                                   "\n\n" @
                                   gappaImplication.ssaPart @ "\n\n" @
                                   gappa.ssa @ "\n" @
                                   ( "poly_res  = " @ prefix @ "tmp_" @ di(gappa.ssano) @ ";\n" @
                                     "Mpoly_res = " @ prefix @ "Mtmp_" @ di(gappa.ssano) @ ";\n" @
                                     "epsilon   = " @ prefix @ "epsilon_" @ di(gappa.ssano) @ ";\n"
                                   ) @
                                   "\n\n" @
                                   "{(\n" @
                                   gappaImplication.implicationPartHypotheses @ "\n" @
                                   ")\n->\n(" @
                                   "\n" @
                                   gappaImplication.implicationPartEpsilon @ "\n" @
                                   ")}\n" @
                                   "\n\n" @
                                   gappa.hints @
                                   gappaImplication.hints @
                                   "\n\n" @
                                   "epsilon $ x in (" @ splitpoints @ ");" @
                                   "\n"
                                 ),
                         .poly = gappa.poly,
                         .powerings = __generateGappaRewritePowerings(prefix, powerings)
                      };
             };
          };

          return res;
};
