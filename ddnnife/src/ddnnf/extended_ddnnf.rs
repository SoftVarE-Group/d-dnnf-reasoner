pub(crate) mod objective_function;
pub(crate) mod optimal_configs;

use crate::ddnnf::anomalies::t_wise_sampling::Config;
use crate::ddnnf::extended_ddnnf::objective_function::ObjectiveFn;
use crate::ddnnf::extended_ddnnf::Attribute::{BoolAttr, FloatAttr, IntegerAttr, StringAttr};
use crate::ddnnf::extended_ddnnf::AttributeValue::{BoolVal, FloatVal, IntegerVal, StringVal};
use crate::Ddnnf;
use itertools::Itertools;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ExtendedDdnnf {
    pub(crate) ddnnf: Ddnnf,
    pub(crate) attrs: HashMap<String, Attribute>,
    pub(crate) objective_fn_vals: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct AttributeT<T> {
    vals: Vec<Option<T>>,
    default_val: Option<T>,
}

impl<T> AttributeT<T> {
    pub fn get_val(&self, var: u32) -> Option<&T> {
        self.vals[(var - 1) as usize]
            .as_ref()
            .or(self.default_val.as_ref())
    }
}

#[derive(Clone, Debug)]
pub enum Attribute {
    IntegerAttr(AttributeT<i32>),
    FloatAttr(AttributeT<f64>),
    BoolAttr(AttributeT<bool>),
    StringAttr(AttributeT<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    IntegerVal(i32),
    FloatVal(f64),
    BoolVal(bool),
    StringVal(String),
}

impl Attribute {
    pub fn get_val(&self, var: u32) -> Option<AttributeValue> {
        match self {
            IntegerAttr(attr) => attr.get_val(var).map(|val| IntegerVal(*val)),
            FloatAttr(attr) => attr.get_val(var).map(|val| FloatVal(*val)),
            BoolAttr(attr) => attr.get_val(var).map(|val| BoolVal(*val)),
            StringAttr(attr) => attr.get_val(var).map(|val| StringVal(val.clone())),
        }
    }

    pub fn new_integer_attr(values: Vec<Option<i32>>, default_val: Option<i32>) -> Self {
        IntegerAttr(AttributeT {
            vals: values,
            default_val,
        })
    }

    pub fn new_float_attr(values: Vec<Option<f64>>, default_val: Option<f64>) -> Self {
        FloatAttr(AttributeT {
            vals: values,
            default_val,
        })
    }

    pub fn new_bool_attr(values: Vec<Option<bool>>, default_val: Option<bool>) -> Self {
        BoolAttr(AttributeT {
            vals: values,
            default_val,
        })
    }

    pub fn new_string_attr(values: Vec<Option<String>>, default_val: Option<String>) -> Self {
        StringAttr(AttributeT {
            vals: values,
            default_val,
        })
    }
}

impl Default for ExtendedDdnnf {
    fn default() -> Self {
        let ddnnf = Ddnnf::default();
        let attrs = HashMap::new();

        ExtendedDdnnf {
            ddnnf,
            attrs,
            objective_fn_vals: None,
        }
    }
}

impl ExtendedDdnnf {
    pub fn new(ddnnf: Ddnnf, attributes: HashMap<String, Attribute>) -> Self {
        ExtendedDdnnf {
            ddnnf,
            attrs: attributes,
            objective_fn_vals: None,
        }
    }

    pub fn get_attr_val(&self, attr_name: &str, var: u32) -> Option<AttributeValue> {
        if let Some(attr) = self.attrs.get(attr_name) {
            return attr.get_val(var);
        }
        None
    }

    pub fn get_objective_fn_val(&self, var: u32) -> f64 {
        match &self.objective_fn_vals {
            Some(vals) => vals[(var - 1) as usize],
            None => panic!("Objective function values have not been calculated."),
        }
    }

    pub fn get_objective_fn_val_of_literals(&self, literals: &[i32]) -> f64 {
        literals
            .iter()
            .map(|&literal| {
                if literal < 0 {
                    0.0
                }
                // deselected
                else {
                    self.get_objective_fn_val(literal.unsigned_abs())
                }
            })
            .sum()
    }

    pub fn get_objective_fn_val_of_config(&self, config: &Config) -> f64 {
        let literals = config.get_decided_literals().collect_vec();
        self.get_objective_fn_val_of_literals(&literals[..])
    }

    pub fn get_average_objective_fn_val_of_literals(&self, literals: &[i32]) -> f64 {
        self.get_objective_fn_val_of_literals(literals) / literals.len() as f64
    }

    pub fn get_average_objective_fn_val_of_config(&self, config: &Config) -> f64 {
        let literals = config.get_decided_literals().collect_vec();
        self.get_average_objective_fn_val_of_literals(&literals[..])
    }

    pub fn calc_objective_fn_vals(&mut self, objective_fn: &ObjectiveFn) {
        let vals = (1..=self.ddnnf.number_of_variables)
            .map(|var| objective_fn.eval(var, &self.attrs))
            .collect_vec();
        self.objective_fn_vals = Some(vals);
    }

    pub fn merge_sorted_configs<'a>(
        &self,
        left: Vec<&'a Config>,
        right: Vec<&'a Config>,
    ) -> Vec<&'a Config> {
        let mut sorted_configs = Vec::with_capacity(left.len() + right.len());
        let mut left_idx = 0;
        let mut right_idx = 0;

        while left_idx < left.len() && right_idx < right.len() {
            if self.get_average_objective_fn_val_of_config(left[left_idx])
                >= self.get_average_objective_fn_val_of_config(right[right_idx])
            {
                sorted_configs.push(left[left_idx]);
                left_idx += 1;
            } else {
                sorted_configs.push(right[right_idx]);
                right_idx += 1;
            }
        }

        while left_idx < left.len() {
            sorted_configs.push(left[left_idx]);
            left_idx += 1;
        }

        while right_idx < right.len() {
            sorted_configs.push(right[right_idx]);
            right_idx += 1;
        }

        sorted_configs
    }

    pub fn insert_config_sorted(&self, config: Config, sorted_configs: &mut Vec<Config>) {
        let config_val = self.get_average_objective_fn_val_of_config(&config);

        let mut curr_idx = sorted_configs.len();
        sorted_configs.push(config);

        while curr_idx > 0
            && config_val > self.get_average_objective_fn_val_of_config(&sorted_configs[curr_idx])
        {
            sorted_configs.swap(curr_idx, curr_idx - 1);
            curr_idx -= 1;
        }
    }

    pub fn are_configs_sorted(&self, configs: Vec<&Config>) -> bool {
        let mut curr_val = f64::MAX;
        for config in configs {
            let next_val = self.get_average_objective_fn_val_of_config(config);
            if next_val > curr_val {
                return false;
            }
            curr_val = next_val;
        }

        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::{build_attributes, build_ddnnf};
    use std::path::Path;

    pub fn build_sandwich_ext_ddnnf() -> ExtendedDdnnf {
        let ddnnf = build_ddnnf(Path::new("tests/data/sandwich.nnf"), Some(19));
        let attributes = build_attributes(Path::new("tests/data/sandwich_attribute_vals.csv"));
        ExtendedDdnnf::new(ddnnf, attributes)
    }

    #[test]
    fn building_and_reading_attributes() {
        let ext_ddnnf = build_sandwich_ext_ddnnf();

        assert_eq!(
            ext_ddnnf.get_attr_val("Calories", 3).unwrap(),
            IntegerVal(203)
        ); // Full Grain
        assert_eq!(ext_ddnnf.get_attr_val("Price", 14).unwrap(), FloatVal(0.99)); // Ham
        assert_eq!(
            ext_ddnnf.get_attr_val("Organic Food", 11).unwrap(),
            BoolVal(false)
        ); // Cream Cheese
    }

    #[test]
    fn calculation_of_objective_function_values() {
        use self::objective_function::{Condition::*, ObjectiveFn::*};

        let mut ext_ddnnf = build_sandwich_ext_ddnnf();
        let objective_fn = IfElse(
            And(
                BoolVar("Organic Food".to_string()).boxed(),
                LessThan(NumVar("Calories".to_string()).boxed(), 10.0).boxed(),
            )
            .boxed(),
            Mul(
                NumVar("Calories".to_string()).boxed(),
                NumConst(100.0).boxed(),
            )
            .boxed(),
            Neg(NumVar("Price".to_string()).boxed()).boxed(),
        );

        let expected = vec![
            0.0,   // Sandwich
            0.0,   // Bread
            -1.99, // Full Grain
            -1.79, // Flatbread
            -1.79, // Toast
            0.0,   // Cheese
            0.0,   // Gouda
            -0.49, // Sprinkled
            -0.69, // Slice
            -0.69, // Cheddar
            -0.59, // Cream Cheese
            0.0,   // Meat
            -1.29, // Salami
            -0.99, // Ham
            -1.39, // Chicken Breast
            0.0,   // Vegetables
            200.0, // Cucumber
            300.0, // Tomatoes
            200.0, // Lettuce
        ];

        ext_ddnnf.calc_objective_fn_vals(&objective_fn);

        assert_eq!(ext_ddnnf.objective_fn_vals.unwrap(), expected);
    }
}
