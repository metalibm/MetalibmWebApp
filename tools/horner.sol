procedure polynomialToTree(p) {
	  var deg, res, poly;
	  var q, r, Q, R, IKS, T;
	  var k, goon;

	  res = { .okay = false };

	  deg = degree(p);
	  if (deg >= 0) then {
	     if (deg == 0) then {
	     	poly = { .type = "constant", .c = coeff(p, 0), .p = p };
		res.okay = true;
	        res.poly = poly;
	     } else {
	        q = coeff(p, 0);
		if (mod(p - q, _x_) == 0) then {
		   r = div(p - q, _x_);
		   k = 1;
		   goon = true;
		   while (goon) do {
		   	 c = coeff(r, 0);
			 if (c == 0) then {
			    if (mod(r, _x_) == 0) then {
			       r = div(r, _x_);
			       k = k + 1;
			       goon = true;
			    } else {
			       goon = false;
			    };
                         } else {
                            goon = false;
                         };
                   };
		   if (q != 0) then {
                      Q = polynomialToTree(q);
		   } else {
		      Q = { .okay = true };
		   };
                   R = polynomialToTree(r);
                   if (Q.okay && R.okay && (k >= 1)) then {
                      if (k == 1) then {
                         IKS = { .type = "variable", .p = _x_ };
                      } else {
		         IKS = { .type = "power", .k = k, .p = _x_^k };
		      };
		      T = { .type = "multiplication", .q = IKS, .r = R.poly, .p = _x_ * r };
		      if (q != 0) then {
		      	 poly = { .type = "addition", .q = Q.poly, .r = T, .p = p };
	              } else {
                         poly = T;
		      };
		      res.okay = true;
	              res.poly = poly;
		   };
		};
	     };
	  };

	  return res;
};

