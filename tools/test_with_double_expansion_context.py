import sollya, polynomial2tree

# ...
sollya.execute("doubleexpansion.sol")
sollya.execute("polynomials.sol")

compareFormats = sollya.parse("compareFormats")
computeActualAdditionBound = sollya.parse("computeActualAdditionBound")
computeActualMultiplicationBound = sollya.parse("computeActualMultiplicationBound")
computeBoundPower = sollya.parse("computeBoundPower")
computeErrorDueToVariableRounding = sollya.parse("computeErrorDueToVariableRounding")
computeOutputFormatAddition = sollya.parse("computeOutputFormatAddition	")
computeOutputFormatMultiplication = sollya.parse("computeOutputFormatMultiplication")
computeOutputFormatPower = sollya.parse("computeOutputFormatPower")
computeFormatForConstant = sollya.parse("computeFormatForConstant")
computeNeededVariableFormat = sollya.parse("computeNeededVariableFormat")
roundConstantToTargetPrecFormatSpecific = sollya.parse("roundConstantToTargetPrecFormatSpecific")

generateGappa = sollya.parse("generateGappa")

class SollyaContext:
  def __init__(self):
    self.variableFormat = 102
    self.compareFormats = compareFormats
    self.computeBoundAddition = computeActualAdditionBound
    self.computeBoundMultiplication = computeActualMultiplicationBound
    self.computeBoundPower = computeBoundPower
    self.computeBoundVariableRounding = computeErrorDueToVariableRounding
    self.computeOutputFormatAddition = computeOutputFormatAddition	
    self.computeOutputFormatMultiplication = computeOutputFormatMultiplication
    self.computeOutputFormatPower = computeOutputFormatPower
    self.computeConstantFormat = computeFormatForConstant
    self.computeVariableFormat = computeNeededVariableFormat
    self.roundConstant = roundConstantToTargetPrecFormatSpecific
  def convert2SollyaObject(self):
    return sollya.SollyaObject({"variableFormat": self.variableFormat, \
                                "compareFormats": self.compareFormats, \
                                "computeBoundAddition": self.computeBoundAddition, \
                                "computeBoundMultiplication": self.computeBoundMultiplication, \
                                "computeBoundPower": self.computeBoundPower, \
                                "computeBoundVariableRounding": self.computeBoundVariableRounding, \
                                "computeOutputFormatAddition": self.computeOutputFormatAddition, \
                                "computeOutputFormatMultiplication": self.computeOutputFormatMultiplication, \
                                "computeOutputFormatPower": self.computeOutputFormatPower, \
                                "computeConstantFormat": self.computeConstantFormat, \
                                "computeVariableFormat": self.computeVariableFormat, \
                                "roundConstant": self.roundConstant})

# ...
#f = sollya.parse("exp(x)")
#I = sollya.parse("[-2^-4;2^-4]")
#n = sollya.parse("9");

#F = [sollya.doubledouble, sollya.doubledouble]+[sollya.binary64]*(n-1);

#p = sollya.fpminimax(f, n, F, I)

f = sollya.parse("sin(x)")
I = sollya.parse("[-2^-4;2^-4]")
Itmp = sollya.parse("[2^-100;2^-4]")
monomials = sollya.parse("[|1, 3, 5, 7, 9, 11|]")

sollya.settings.prec = 3000;
F = [sollya.doubledouble, sollya.doubledouble]+[sollya.binary64]*4;

p = sollya.fpminimax(f, monomials, F, Itmp);
sollya.settings.prec = 165;

epsTarget = sollya.dirtyinfnorm(p/f-1, I)

# ...
computeErrorBounds = sollya.parse("computeErrorBounds")
context = SollyaContext().convert2SollyaObject()
PR, powers = polynomial2tree.buildTree(p, power=2)
PR = PR.convert2SollyaObject()

if PR["okay"] == True:
  P = PR["poly"]
  R = computeErrorBounds(P, I, epsTarget, context)
  print("result= ", R)

Gappa = generateGappa("foo_", R.struct.poly, R.struct.powerings, context)

if Gappa.struct.okay:
    print(Gappa.struct.gappa)
    with open("gappa.gappa", "w") as gappa_proof:
        gappa_proof.write(str(Gappa.struct.gappa))
else:
    print("Error: failed to generate gappa proof")
