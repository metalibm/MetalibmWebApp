import itertools
import cgpe, sollya

def get_power(scheme):
    if isinstance(scheme, cgpe.Variable):
        return 1
    elif isinstance(scheme, cgpe.Multiplication):
        pQ = get_power(scheme.op1)
        pR = get_power(scheme.op2)
        if pQ != None and pR != None:
            return pQ + pR
    return None
    
# ...
class PolyTree:
    def __init__(self, scheme, poly, power, first_coeff, reduce_power = True, expression = None, eliminate_common_subexpression = True):
        expression = {} if expression is None else expression
        # print("PolyTree constructor called with scheme = ", scheme)
        self.p = None
        self.q = None
        self.r = None
        self.type = None
        # self.ops = None
        self.name = None # GR: useless, just for testing
        self.k = None
        # print("scheme: {}".format(scheme))
        self.latency = scheme.latency
        #
        if isinstance(scheme, cgpe.Variable):
            self.p = sollya.parse("_x_") # ^"+str(power))
            # self.ops = 1
            #if power == 1:
            self.type = "variable"
            self.name = 'x'
            #else:
            #    self.type = "power"
            #    self.name = 'x['+str(power)+']'
            #    self.k = power
            expression[str(scheme)] = self
        elif isinstance(scheme, cgpe.Constant):
            num = int(scheme.name[1:])*power+first_coeff
            # print("Extracting coefficient num = ", num, " out of poly = ", poly)
            self.p = sollya.coeff(poly, num)
            # self.ops = 1
            self.type = "constant"
            self.name = scheme.name[0]+str(num)
            expression[str(scheme)] = self
        elif isinstance(scheme, cgpe.Addition) or isinstance(scheme, cgpe.Multiplication):
            self.k = get_power(scheme)
            if self.k != None and reduce_power == True:
                self.p = sollya.parse("_x_^"+str(self.k))
                # self.ops = 1
                self.type = "power"
                self.name = 'x['+str(self.k)+']'
                expression[str(scheme)] = self
            else:
                # ... operand1: Q
                if str(scheme.op1) in expression and eliminate_common_subexpression:
                    self.q = expression[str(scheme.op1)]
                else:
                    self.q = PolyTree(scheme.op1, poly, power, first_coeff, reduce_power, expression, eliminate_common_subexpression)
                # ... operand2: R
                if str(scheme.op2) in expression and eliminate_common_subexpression:
                    self.r = expression[str(scheme.op2)]
                elif scheme.op1 != scheme.op2 and str(scheme.op1) != str(scheme.op2): # GR: argh, this is ugly !
                    self.r = PolyTree(scheme.op2, poly, power, first_coeff, reduce_power, expression, eliminate_common_subexpression)
                else:
                    self.r = self.q
                # ...
                # self.ops = self.q.ops + self.r.ops +1
                if isinstance(scheme, cgpe.Addition):
                    self.p = self.q.p + self.r.p                
                    self.type = "addition"
                elif isinstance(scheme, cgpe.Multiplication):
                    self.p = self.q.p * self.r.p                
                    self.type = "multiplication"
            expression[str(scheme)] = self

    def __str__(self):
        if self.type == 'addition':
            return '({}+{})'.format(self.q, self.r)
        elif self.type == 'multiplication':
            return '({}*{})'.format(self.q, self.r)
        else:
            return self.name

    def convert2SollyaObject(self):
        res = {"okay": False, "poly": None}
        poly = {"type": self.type, "p": self.p} #, "ops": self.ops}
        if self.type == "constant":
            poly["c"] = self.p
            res["okay"] = True
            res["poly"] = poly
        elif self.type == "variable":
            res["okay"] = True
            res["poly"] = poly
        elif self.type == "power":
            poly["k"] = self.k
            res["okay"] = True
            res["poly"] = poly
        else:
            Q = self.q.convert2SollyaObject()
            R = self.r.convert2SollyaObject()
            if Q["okay"] == True and R["okay"] == True:
                poly["q"] = Q["poly"]
                poly["r"] = R["poly"]
                res["okay"] = True
                res["poly"] = poly
        return res
        #

def update_latency(scheme, driver):
        if isinstance(scheme, cgpe.Multiplication):
            return max(scheme.op1.latency, scheme.op2.latency) + driver.get_multiplier_latency()
        elif isinstance(scheme, cgpe.Addition):
            return max(scheme.op1.latency, scheme.op2.latency) + driver.get_adder_latency()
        elif scheme != None:
            return scheme.latency

def simplify(scheme, p, power, first_coeff, driver):
    new_latency = scheme.latency
    if scheme != None and not(isinstance(scheme, cgpe.Variable)) and not(isinstance(scheme, cgpe.Constant)):
        # ... let's go inside
        scheme.op1, scheme.op1.latency = simplify(scheme.op1, p, power, first_coeff, driver)
        scheme.op2, scheme.op2.latency = simplify(scheme.op2, p, power, first_coeff, driver)
        # ... let's simplify current node
        if isinstance(scheme, cgpe.Addition) and isinstance(scheme.op1, cgpe.Constant):
            num = int(scheme.op1.name[1:]) * power + first_coeff
            val = sollya.coeff(p, num)
            if val == 0:
                return scheme.op2, scheme.op2.latency
        elif isinstance(scheme, cgpe.Addition) and isinstance(scheme.op2, cgpe.Constant):
            num = int(scheme.op2.name[1:]) * power + first_coeff
            val = sollya.coeff(p, num)
            if val == 0:
                return scheme.op1, scheme.op1.latency
        elif isinstance(scheme, cgpe.Multiplication) and isinstance(scheme.op1, cgpe.Constant):
            num = int(scheme.op1.name[1:]) * power + first_coeff
            val = sollya.coeff(p, num)
            if val == 0:
                return scheme.op1, scheme.op1.latency
            if val == 1:
                return scheme.op2, scheme.op2.latency
        elif isinstance(scheme, cgpe.Multiplication) and isinstance(scheme.op2, cgpe.Constant):
            num = int(scheme.op2.name[1:]) * power + first_coeff
            val = sollya.coeff(p, num)
            if val == 0:
                return scheme.op2, scheme.op2.latency
            if val == 1:
                return scheme.op1, scheme.op1.latency
        # ... update latency
        new_latency = update_latency(scheme, driver)
    return scheme, new_latency
        
def expand(scheme, powers, driver):
    if scheme != None and not(isinstance(scheme, cgpe.Variable)) and not(isinstance(scheme, cgpe.Constant)):
        scheme.op1 = expand(scheme.op1, powers, driver)
        scheme.op2 = expand(scheme.op2, powers, driver)
        if isinstance(scheme.op1, cgpe.Variable):
            scheme.op1 = powers
        if isinstance(scheme.op2, cgpe.Variable):
            scheme.op2 = powers
        # ... update latency
        scheme.latency = update_latency(scheme, driver)
    return scheme

def determinePowers(scheme, exprs):
    if scheme != None and (isinstance(scheme, cgpe.Addition) or isinstance(scheme, cgpe.Multiplication)):
        p1 = determinePowers(scheme.op1, exprs)
        p2 = determinePowers(scheme.op2, exprs)
        if isinstance(scheme, cgpe.Multiplication):
            p = get_power(scheme)
            if p != None:
                if p not in exprs.keys():
                    exprs[p] = scheme.latency
                else:
                    exprs[p] = min(exprs[p], scheme.latency)
                return p1+p2+[scheme]
        return p1+p2
    return []

def extractPowers(tree):
    if tree != None and tree.type == "power":
        return [tree.k]
    elif tree != None and tree.type in ["addition", "multiplication"]:
        p1 = extractPowers(tree.q)
        p2 = extractPowers(tree.r)
        p = list(set(p1+p2))
        p.sort()
        return p
    return []

def number_multiplier(scheme, subexpressions):
    if isinstance(scheme, cgpe.Multiplication):
        if str(scheme) in subexpressions:
            return 0
        m1 = number_multiplier(scheme.op1, subexpressions)
        m2 = number_multiplier(scheme.op2, subexpressions)
        subexpressions.append(str(scheme))
        return m1+m2+1
    return 0

def factorize_x(scheme, driver):
    if scheme != None and not(isinstance(scheme, cgpe.Variable)) and not(isinstance(scheme, cgpe.Constant)):
        scheme.op1 = factorize_x(scheme.op1, driver)
        scheme.op2 = factorize_x(scheme.op2, driver)
        if isinstance(scheme, cgpe.Multiplication) and get_power(scheme.op1) != None and isinstance(scheme.op2, cgpe.Multiplication) and get_power(scheme.op2.op1) != None:
            var_power = scheme.op1
            current_power = get_power(scheme.op2.op1)
            while(current_power >= 1):
                var_power = cgpe.Multiplication(cgpe.Variable(name = "x"), var_power, driver)
                current_power -= 1
            scheme.op1 = var_power
            scheme.op2 = scheme.op2.op2
        # ... update latency
        scheme.latency = update_latency(scheme, driver)
    return scheme

def estimate_power_latency(k, mul):
    if k == 1:
        return 0
    else:
        if k % 2 == 0:
            return mul + estimate_power_latency(k//2, mul)
        else:
            return mul + max(estimate_power_latency(k//2, mul), estimate_power_latency(k//2+1, mul))

def buildTree(p, classic = None, power = False, driver = None):
    """
        Returns a polynomial tree, together with trees for evaluating powers whose evaluation reduce the number of multiplications.
    """
    if driver == None:
        driver = cgpe.CgpeDriver()
    d = sollya.degree(p)
    # ... this checking step should be useless one day...
    starting_coef = 0
    while(starting_coef <= d and sollya.coeff(p, starting_coef) == 0):
        starting_coef += 1
    # ...
    x_k = 1
    if starting_coef != d:
        x_k = 2
        found = False
        while x_k <= 10 and found == False: # GR: this is ugly, and '10' is experimental
            power_is_ok = [True if (i + starting_coef) % x_k == 0 else sollya.coeff(p, i) == 0 for i in range(int(d+1))]
            if len(set(power_is_ok)) == 1 and power_is_ok[0] != False:
                found = True
            else:
                x_k = x_k + 1
        if found == False:
            starting_coef = 0
            x_k = 1
        elif x_k > 1:
            d = ((d - starting_coef) / x_k)
    else:
        starting_coef = 0
        x_k = 1
    # ...
    coefficient_list = [sollya.coeff(p, starting_coef + i * x_k) != 0 for i in range(int(d + 1))]
    if classic == None:
        scheme = driver.get_low_latency_scheme(d, coefficient_list)
    elif classic == 'horner':
        scheme = driver.get_horner_scheme(d)
    elif classic == 'horner2':
        scheme = driver.get_horner_scheme(d, 2)
    elif classic == 'horner4':
        scheme = driver.get_horner_scheme(d, 4)
    elif classic == 'estrin':
        scheme = driver.get_estrin_scheme(d)
    if x_k > 1:
        # ... expand
        var_power = cgpe.Variable(name = "x")
        current_power = x_k - 1;
        while(current_power >= 1):
            var_power = cgpe.Multiplication(cgpe.Variable(name = "x"), var_power, driver)
            current_power -= 1
        expanded_scheme = expand(scheme.root, var_power, driver)
        # ... add last multiplication
        if starting_coef != 0:
            var_power = cgpe.Variable(name = "x")
            current_power = starting_coef - 1;
            while(current_power >= 1):
                var_power = cgpe.Multiplication(cgpe.Variable(name = "x"), var_power, driver)
                current_power -= 1
            scheme = cgpe.PolynomialScheme(cgpe.Multiplication(var_power, expanded_scheme, driver), driver)
        else:
            scheme = cgpe.PolynomialScheme(expanded_scheme, driver)
    #
    scheme.root, scheme.latency = simplify(scheme.root, p, x_k, starting_coef, driver)
    factorize_x(scheme.root, driver)
    #
    tree = PolyTree(scheme.root, p, x_k, starting_coef, eliminate_common_subexpression = False)
    #
    print(" *** expression=", tree)
    # print(" *** poly=", tree.p)
    # ...
    # exprs = {}
    # schemes = determinePowers(scheme.root, exprs)
    # degrees = list(exprs.keys())
    degrees = extractPowers(tree)
    # degrees.sort()
    # ...
    expressions = []
    for d in degrees:
        p_driver = cgpe.CgpeDriver(adder_latency = driver.get_adder_latency(), multiplier_latency = driver.get_multiplier_latency())
        target_latency = driver.get_multiplier_latency() + estimate_power_latency(d, driver.get_multiplier_latency())
        expressions.append(p_driver.get_power_expressions(d, latency=target_latency)) # scheme.root.latency)) # exprs[d]))
        del p_driver
    # ...
    best_combination = []
    best_muls = 0
    for index, e in enumerate(itertools.product(*expressions)):
        subexprs = []
        muls = 0
        for i in e:
            muls += number_multiplier(i.root, subexprs)
        if len(best_combination) == 0 or (best_muls > muls) or (best_muls == muls and best_combination[len(best_combination)-1].root.latency > e[len(e)-1].root.latency):
            best_combination = e[:]
            best_muls = muls
    best_trees = [PolyTree(e.root, None, None, None, reduce_power = False, eliminate_common_subexpression = False) for e in best_combination]
    # ...
    del driver
    # ...
    return tree, dict(zip(degrees, best_trees))


if __name__ == "__main__":
   p = sollya.parse(" x * (0.99999999999999999999999708926504228497151835134763 + x^2 * (0.16666666666666666666752458486937403871413487479233 + x^2 * (8.3333333333333332985867855416919593920153920267693e-3 + x^2 * (1.9841269841269893442090514962631403580854930972846e-4 + x^2 * (2.7557319223946215620568294926669829836373569175431e-6 + x^2 * (2.50521084034411073262250912981854639268155232657234e-8 + x^2 * (1.6059038222232095750091066523224572233905214002566e-10 + x^2 * (7.6489783889677795142833657355836262246495872032567e-13 + x * (2.3593431615962715576327153258082641875215186794959e-16 + x * 2.9157922781061909021477344217566969054557572947271e-15)))))))))")
   PR, powers = buildTree(p, classic="estrin", power=True)
