import string
import sollya


from metalibm_core.core.ml_formats import (
    ML_Binary32,
    ML_Binary64, ML_DoubleDouble, ML_TripleDouble
)
from metalibm_core.core.ml_operations import (
    Addition, Multiplication, Constant, Variable, Conversion,
)

from metalibm_core.utility.debug_utils import debug_multi

def expand_node(node):
    return node

def tag_node(tag, node, **extra_attributes):
    """ from SollyaObject @p poly extract name tag and
        apply it to expanded_tuple limbs """
    node.set_tag("{}".format(tag))
    node.set_attributes(**extra_attributes)
    return node

def tag_node_from_poly(poly, node, **extra_attributes):
    tag = poly.struct.name
    return tag_node(tag, node, **extra_attributes)


class SollyaHDLConverter:
    """ Sollya to metalibm converter for HDL operation,
        including fixed-point support """
    def __init__(self, root_var, lutetia_tag_mode=False):
        self.root_var = root_var
        self.power_table = {}
        self.name_map = {}
        self.letter_set = set()
        for c in string.ascii_uppercase:
            for b in string.ascii_uppercase:
                for a in string.ascii_uppercase:
                    self.letter_set.add(a + b + c)

        self.lutetia_tag_mode = lutetia_tag_mode

    def get_name(self, node, as_input=True):
        if node in self.name_map:
            return self.name_map[node]
        else:
            return "undef"

    def parse_sollya_format(self, sollya_format):
        """ convert a format described in a ml_precision sollya structure
            into a metalibm format object """
        # TODO: for now discards accuracy info
        py_format = sollya_format.python()
        return py_format

    def get_new_letter(self):
        return self.letter_set.pop()

    def tag_new_op(self, poly, op_list, desc="UNK"):
        if self.lutetia_tag_mode:
            print("poly={}".format(str(poly)))
            return poly.struct.name
        else:
            out_tag = self.get_new_letter()
            tag = "{}_{}_{}".format(
                out_tag,
                desc,
                "_".join(self.get_name(op) for op in op_list)
            )
            self.name_map[poly] = out_tag
            return tag

    def parse_sollya_addition(self, poly):
        """ generate a Metalibm Addition node """
        ml_format = self.parse_sollya_format(poly.struct.format)
        lhs = self.parse_sollya_poly(poly.struct.q)
        rhs = self.parse_sollya_poly(poly.struct.r)
        tag = self.tag_new_op(poly, [poly.struct.q, poly.struct.r], "Add")
        return tag_node(tag, Addition(lhs, rhs, precision=ml_format.fixed_format), debug=debug_multi)

    def parse_sollya_multiplication(self, poly):
        """ generate a Metalibm Multiplication node """
        ml_format = self.parse_sollya_format(poly.struct.format)
        lhs = self.parse_sollya_poly(poly.struct.q)
        rhs = self.parse_sollya_poly(poly.struct.r)
        tag = self.tag_new_op(poly, [poly.struct.q, poly.struct.r], "Mul")
        return tag_node(tag, Multiplication(lhs, rhs, precision=ml_format.fixed_format), debug=debug_multi)

    def parse_sollya_constant(self, cst):
        """ generate a Metalibm Constant node """
        value = cst.struct.c
        ml_format = self.parse_sollya_format(cst.struct.format)
        full_cst = Constant(value, precision=ml_format.fixed_format)
        result = full_cst
        tag = self.tag_new_op(cst, [], "Cst")
        #return tag_node_from_poly(cst, result)
        return tag_node(tag, result, debug=debug_multi)

    def parse_sollya_variable(self, var):
        """ generate metalibm Variable node from sollya object,
            ensuring input variable node unicity """
        ml_format = self.parse_sollya_format(var.struct.format)
        input_precision = self.parse_sollya_format(var.struct.inputformat).fixed_format
        if ml_format.fixed_format is input_precision:
            result = expand_node(self.root_var)
            tag = self.root_var.get_tag()
        else:
            result = expand_node(Conversion(
                self.root_var,
                precision=ml_format.fixed_format,
                tag=var.struct.name
            ))
            # only tag converted variabe as we do not want to modify
            # the tag associated to an input Variable
            tag = self.tag_new_op(result, [], "Var")
        return tag_node(tag, result)


    def get_power(self, k, ml_format):
        epsilon = ml_format.rel_error
        precision = ml_format.fixed_format
        result = None
        if k == 1:
            if precision is self.root_var.precision:
                result = self.root_var
            else:
                result = Conversion(self.root_var, precision=precision, rel_error=epsilon)
        elif k in self.power_table:
            possible_nodes = []
            for power_format, power_node, power_epsilon in self.power_table[k]:
                if power_format == precision and power_epsilon <= epsilon:
                    possible_nodes.append((power_format, power_node, power_epsilon))
            if len(possible_nodes) == 0:
                for power_format, power_node, power_epsilon in self.power_table[k]:
                    if power_epsilon <= epsilon:
                        possible_nodes.append((power_format, power_node, power_epsilon))
            result_format = min(possible_nodes, key=(lambda v: v[2]))[0]
            result = min(possible_nodes, key=(lambda v: v[2]))[1]
            if result_format != precision:
                result = Conversion(
                    result, precision=power_format,
                )
        else:
            print("k={} not found in self.power_table".format(k))
            raise Exception()
        assert not result is None
        return result

    def parse_sollya_power(self, power):
        """ parsing sollya power:
            self.root_var ^ power.struct.k """
        k = int(power.struct.k)
        ml_format = self.parse_sollya_format(power.struct.format)

        if self.lutetia_tag_mode:
            return tag_node_from_poly(power, self.get_power(k, ml_format))
        else:
            tag = "pow{}".format(k)
            self.name_map[power] = tag
            return tag_node(tag, self.get_power(k, ml_format), debug=debug_multi)

    def parse_sollya_poly(self, poly):
        """ parsing sollya's polynomial object returned by
            computeErrorBounds method """
        parsing_method_map = {
            "addition": self.parse_sollya_addition,
            "multiplication": self.parse_sollya_multiplication,
            "constant": self.parse_sollya_constant,
            "variable": self.parse_sollya_variable,
            "power": self.parse_sollya_power,
        }
        class_key = poly.struct.type
        return parsing_method_map[str(poly.struct.type)](poly)


    def get_power_degree(self, tree):
        if tree != None and tree.struct.type == "variable":
            return 1
        elif tree != None and tree.struct.type == "multiplication":
            return self.get_power_degree(tree.struct.q) + self.get_power_degree(tree.struct.r)
        return 0

    def build_powers_for(self, tree):
        """ build a subgraph to compute x^k """
        result = None
        k = self.get_power_degree(tree)
        ml_format = self.parse_sollya_format(tree.struct.format)
        precision = ml_format.fixed_format
        epsilon = ml_format.rel_error
        exist = False
        #
        if k == 1:
            if precision is self.root_var.precision:
                result = self.root_var
            else:
                result = Conversion(self.root_var, precision=precision)
        elif k > 1:
            if k in self.power_table:
                for power_format, power_node, power_epsilon in self.power_table[k]:
                    if power_format == precision and power_epsilon <= ml_format.rel_error:
                        result = power_node
                        exist = True
                    elif power_epsilon <= ml_format.rel_error:
                        result = Conversion(
                            power_node, precision=power_format,
                        )
            if result == None:
                r1 = self.build_powers_for(tree.struct.q)
                r2 = self.build_powers_for(tree.struct.r)
                result = Multiplication(r1, r2, precision=ml_format.fixed_format)
        else:
            raise Exception("unsupported k value in build_powers_for")
        #
        if exist == False:
            if k in list(self.power_table.keys()):
                self.power_table[k].append((precision, result, epsilon))
            else:
                self.power_table[k] = [(precision, result, epsilon)]
        assert not result is None
        return result

    def build(self, poly, powers):
        for tree in powers:
            self.build_powers_for(tree)
        return self.parse_sollya_poly(poly)

# parser = SollyaConverter()

# sollya.execute("test_mll.sol")
# R = sollya.parse("R")
# print(parser.parse_sollya_poly(R.struct.poly))
