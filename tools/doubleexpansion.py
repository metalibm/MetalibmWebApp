import sollya, polynomial2tree

sollya.execute("doubleexpansion.sol")
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

context = SollyaContext().convert2SollyaObject()
