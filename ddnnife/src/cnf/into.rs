use crate::{Ddnnf, NodeType};
use ddnnife_cnf::{Clause, Cnf, Literal};
use std::collections::HashMap;

type Clauses = Vec<Clause>;

/// An operation of a node.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum OperationType {
    And,
    Or,
}

/// An operation between multiple literals.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Operation {
    /// The operation connecting the literals.
    op_type: OperationType,
    /// The literals connected by the operation.
    literals: Vec<Literal>,
}

impl Operation {
    /// Returns the number of literals in the operation.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Returns whether the operation contains no literals.
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
}

/// A biconditional connecting literals via an operation.
#[derive(Debug, Clone)]
struct Biconditional {
    /// The variable representing this biconditional.
    index: usize,
    /// Which operation is connecting the literals.
    op_type: OperationType,
    /// The literals connected by the operation.
    literals: Vec<Literal>,
}

impl From<Biconditional> for Clauses {
    /// Transforms a biconditional into CNF clauses.
    fn from(biconditional: Biconditional) -> Self {
        // Extract the literal representing the index of the biconditional.
        let tseitin_literal = biconditional.index as Literal;

        // Use a predefined representation for CNF transformation.
        match biconditional.op_type {
            // X ↔ A ∧ B ∧ ... => (X ∨ ¬A ∨ ¬B ∨ ...) ∧ (¬X ∨ A) ∧ (¬X ∨ B) ∧ ...
            OperationType::And => {
                let mut first = Vec::with_capacity(biconditional.literals.len() + 1);
                first.push(tseitin_literal);
                first.extend(biconditional.literals.iter().map(|literal| -literal));

                let mut clauses = Vec::with_capacity(biconditional.literals.len() + 1);
                clauses.push(first);
                clauses.extend(
                    biconditional
                        .literals
                        .iter()
                        .map(|literal| vec![-tseitin_literal, *literal]),
                );

                clauses
            }
            // X ↔ A ∨ B ∨ ... => (¬X ∨ A ∨ B ∨ ...) ∧ (X ∨ ¬A) ∧ (X ∨ ¬B)  ∧ ...
            OperationType::Or => {
                let mut first = Vec::with_capacity(biconditional.literals.len() + 1);
                first.push(-tseitin_literal);
                first.append(&mut biconditional.literals.clone());

                let mut clauses = Vec::with_capacity(biconditional.literals.len() + 1);
                clauses.push(first);
                clauses.extend(
                    biconditional
                        .literals
                        .iter()
                        .map(|literal| vec![tseitin_literal, -literal]),
                );

                clauses
            }
        }
    }
}

impl From<&Ddnnf> for Cnf {
    /// Generates an equi-countable CNF representation via Tseitin transformation.
    ///
    /// **Note:** The d-DNNF is not allowed to contain boolean nodes in positions other than the root
    /// node.
    fn from(ddnnf: &Ddnnf) -> Self {
        // Check for special case of boolean root nodes.
        let root_index = ddnnf.nodes.len() - 1;
        let root = &ddnnf.nodes[root_index];

        // A `true` root node implicates a model count of `1`.
        if root.ntype == NodeType::True {
            return Cnf::with_count_1();
        }

        // A `false` root node implicates a model count of `0`.
        if root.ntype == NodeType::False {
            return Cnf::with_count_0();
        }

        // Index to keep track of the last variable introduced to reference a Tseitin transformation.
        let mut tseitin_index = ddnnf.number_of_variables as usize + 1;

        // Biconditionals representing
        let mut biconditionals = Vec::new();

        // Literals representing each node.
        // Literals stay the same, operation nodes get a new variable referencing their transformation assigned.
        let mut node_literals = vec![0; ddnnf.nodes.len()];

        // Cache for already transformed operations.
        let mut cache = HashMap::new();

        ddnnf.nodes.iter().enumerate().for_each(|(index, node)| {
            let literal = match &node.ntype {
                NodeType::And { children } => transform_operation(
                    Operation {
                        op_type: OperationType::And,
                        literals: nodes_to_literals(children, &node_literals),
                    },
                    &mut biconditionals,
                    &mut cache,
                    &mut tseitin_index,
                ),
                NodeType::Or { children } => transform_operation(
                    Operation {
                        op_type: OperationType::Or,
                        literals: nodes_to_literals(children, &node_literals),
                    },
                    &mut biconditionals,
                    &mut cache,
                    &mut tseitin_index,
                ),
                NodeType::Literal { literal } => *literal as Literal,
                NodeType::True => unreachable!("Boolean nodes are only allowed as a root node."),
                NodeType::False => unreachable!("Boolean nodes are only allowed as a root node."),
            };

            node_literals[index] = literal;
        });

        // In case only literals were processed, return an empty CNF.
        if tseitin_index == ddnnf.number_of_variables as usize + 1 {
            return Cnf::default();
        }

        let mut cnf: Clauses = biconditionals.into_iter().flat_map(Clauses::from).collect();
        cnf.push(vec![tseitin_index as Literal - 1]);
        cnf.into()
    }
}

fn nodes_to_literals(nodes: &[usize], node_literals: &[Literal]) -> Vec<Literal> {
    nodes.iter().map(|node| node_literals[*node]).collect()
}

/// Turns an operation into biconditionals.
///
/// Panics when the operation contains no literals.
fn transform_operation(
    operation: Operation,
    biconditionals: &mut Vec<Biconditional>,
    cache: &mut HashMap<Operation, usize>,
    tseitin_index: &mut usize,
) -> Literal {
    if operation.is_empty() {
        panic!("Attempt to transform empty operation.");
    }

    if operation.len() == 1 {
        return operation.literals[0] as Literal;
    }

    // Check if this operation was already handled.
    // In this case, the node can be represented by the same Tseitin variable.
    if let Some(&literal) = cache.get(&operation) {
        return literal as Literal;
    }

    // Keep the index representing the current operation and increment the global one.
    let current_index = *tseitin_index;
    *tseitin_index += 1;

    // Introduce a new proposition representing the operation.
    biconditionals.push(Biconditional {
        index: current_index,
        op_type: operation.op_type,
        literals: operation.literals.clone(),
    });

    // Cache the current operation.
    cache.insert(operation, current_index);

    // Return the index representing the root of the operation.
    current_index as Literal
}
