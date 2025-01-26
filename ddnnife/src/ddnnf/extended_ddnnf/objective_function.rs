use self::Condition::*;
use self::ObjectiveFn::*;
use super::Attribute;
use super::Attribute::{BoolAttr, FloatAttr, IntegerAttr, StringAttr};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct FloatOrd {
    val: f64,
}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.val.total_cmp(&other.val)
    }
}

impl std::cmp::Eq for FloatOrd {}

impl FloatOrd {
    pub fn from(val: f64) -> Self {
        Self { val }
    }
}

pub enum ObjectiveFn {
    IfElse(Box<Condition>, Box<ObjectiveFn>, Box<ObjectiveFn>),
    Add(Box<ObjectiveFn>, Box<ObjectiveFn>),
    Sub(Box<ObjectiveFn>, Box<ObjectiveFn>),
    Mul(Box<ObjectiveFn>, Box<ObjectiveFn>),
    Div(Box<ObjectiveFn>, Box<ObjectiveFn>),
    Neg(Box<ObjectiveFn>),
    NumVar(String),
    NumConst(f64),
}

pub enum Condition {
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Xor(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
    Eq(Box<ObjectiveFn>, f64),
    LessThan(Box<ObjectiveFn>, f64),
    LessEq(Box<ObjectiveFn>, f64),
    GreaterThan(Box<ObjectiveFn>, f64),
    GreaterEq(Box<ObjectiveFn>, f64),
    StringMatch(String, Regex),
    BoolVar(String),
    BoolConst(bool),
}

impl ObjectiveFn {
    pub fn eval(&self, var: u32, attrs: &HashMap<String, Attribute>) -> f64 {
        match self {
            IfElse(cond, if_expr, else_expr) => {
                if cond.eval(var, attrs) {
                    if_expr.eval(var, attrs)
                } else {
                    else_expr.eval(var, attrs)
                }
            }
            Add(left_expr, right_expr) => left_expr.eval(var, attrs) + right_expr.eval(var, attrs),
            Sub(left_expr, right_expr) => left_expr.eval(var, attrs) - right_expr.eval(var, attrs),
            Mul(left_expr, right_expr) => left_expr.eval(var, attrs) * right_expr.eval(var, attrs),
            Div(left_expr, right_expr) => left_expr.eval(var, attrs) / right_expr.eval(var, attrs),
            Neg(expr) => expr.eval(var, attrs) * (-1.0),
            NumVar(attr_name) => get_numeric_attr_val_or_panic(attr_name, var, attrs),
            NumConst(val) => *val,
        }
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

impl Condition {
    pub fn eval(&self, var: u32, attrs: &HashMap<String, Attribute>) -> bool {
        match self {
            And(left_cond, right_cond) => left_cond.eval(var, attrs) && right_cond.eval(var, attrs),
            Or(left_cond, right_cond) => left_cond.eval(var, attrs) || right_cond.eval(var, attrs),
            Xor(left_cond, right_cond) => left_cond.eval(var, attrs) ^ right_cond.eval(var, attrs),
            Not(cond) => !cond.eval(var, attrs),
            Eq(expr, cmp_val) => expr.eval(var, attrs) == *cmp_val,
            LessThan(expr, cmp_val) => expr.eval(var, attrs) < *cmp_val,
            LessEq(expr, cmp_val) => expr.eval(var, attrs) <= *cmp_val,
            GreaterThan(expr, cmp_val) => expr.eval(var, attrs) > *cmp_val,
            GreaterEq(expr, cmp_val) => expr.eval(var, attrs) >= *cmp_val,
            StringMatch(attr_name, regex) => {
                regex.is_match(get_string_attr_val_or_panic(attr_name, var, attrs))
            }
            BoolVar(attr_name) => get_bool_attr_val_or_panic(attr_name, var, attrs),
            BoolConst(val) => *val,
        }
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

pub fn get_numeric_attr_val_or_panic(
    attr_name: &str,
    var: u32,
    attrs: &HashMap<String, Attribute>,
) -> f64 {
    match attrs.get(attr_name) {
        Some(IntegerAttr(attr)) => *attr
            .get_val(var)
            .unwrap_or_else(|| panic!("No \'Integer\' value provided for feature {var}."))
            as f64,
        Some(FloatAttr(attr)) => *attr
            .get_val(var)
            .unwrap_or_else(|| panic!("No \'Float\' value provided for feature {var}.")),
        Some(_) => {
            panic!("Attribute {attr_name} is neither of type \'Integer\', nor of type \'Float\'.")
        }
        None => panic!("Attribute {attr_name} does not exist."),
    }
}

pub fn get_bool_attr_val_or_panic(
    attr_name: &str,
    var: u32,
    attrs: &HashMap<String, Attribute>,
) -> bool {
    match attrs.get(attr_name) {
        Some(BoolAttr(attr)) => *attr
            .get_val(var)
            .unwrap_or_else(|| panic!("No \'Bool\' value provided for feature {var}.")),
        Some(_) => panic!("Attribute {attr_name} is not of type \'Bool\'."),
        None => panic!("Attribute {attr_name} does not exist."),
    }
}

pub fn get_string_attr_val_or_panic<'a>(
    attr_name: &str,
    var: u32,
    attrs: &'a HashMap<String, Attribute>,
) -> &'a String {
    match attrs.get(attr_name) {
        Some(StringAttr(attr)) => attr
            .get_val(var)
            .unwrap_or_else(|| panic!("No \'String\' value provided for feature {var}.")),
        Some(_) => panic!("Attribute {attr_name} is not of type \'String\'."),
        None => panic!("Attribute {attr_name} does not exist."),
    }
}
