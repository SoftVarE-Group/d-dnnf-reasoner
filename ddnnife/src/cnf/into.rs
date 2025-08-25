use crate::{Ddnnf, NodeType};
use ddnnife_cnf::{Clause, Cnf, Literal};
use std::{collections::{HashMap, HashSet}, ops::BitOr};

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

impl Biconditional {
    /// The decisions refer to the index the nodes in the list of biconditionals and not(!) to the variable index
    /// Returns 0 if no new decision emerged
    fn decision_propagation(&mut self, decisions: &Vec<Literal>) -> Literal {
        match self.op_type {
            OperationType::And => {
                if self
                    .literals
                    .iter()
                    .any(|literal| decisions.contains(&(-*literal)))
                {
                    return -(self.index as isize);
                }
                self.literals.retain(|literal| decisions.contains(literal));
                if self.literals.is_empty() {
                    return self.index as isize;
                }
            }
            OperationType::Or => {
                if self
                    .literals
                    .iter()
                    .any(|literal| decisions.contains(literal))
                {
                    self.index as isize;
                }
                self.literals
                    .retain(|literal| decisions.contains(&(-*literal)));
                if self.literals.is_empty() {
                    self.index as isize;
                }
            }
        }
        return 0;
    }

    fn is_taut(&self) -> bool {
        let mut is_taut = false;
        match self.op_type {
            OperationType::Or => {
                let mut checked_literals: Vec<Literal> = Vec::new();
                self.literals.iter().for_each(|literal| {
                    if checked_literals.contains(&(-*literal)) {
                        is_taut = true;
                    }
                    checked_literals.push(*literal);
                });
                return is_taut;
            }
            _ => false,
        }
    }

    fn replace_variable_index(&mut self, old: usize, new: usize) {
        if self.index == old {
            self.index = new;
        }
        self.literals.iter_mut().for_each(|lit| {
            if *lit == old as isize {
                *lit = new.clone() as isize;
            }
        });
    }
}

impl From<&mut Biconditional> for Clauses {
    /// Transforms a biconditional into CNF clauses.
    fn from(biconditional: &mut Biconditional) -> Self {
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
    fn from(ddnnf: &Ddnnf) -> Self {
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
                // TODO: what to do about these?
                NodeType::True => unreachable!(),
                // TODO: what to do about these?
                NodeType::False => unreachable!(),
            };

            node_literals[index] = literal;
        });

        // In case only literals were processed, return an empty CNF.
        if tseitin_index == ddnnf.number_of_variables as usize + 1 {
            return Cnf::default();
        }

        let simplified_biconditionals = simplify_biconditionals(ddnnf, &mut biconditionals);

        let mut cnf: Clauses = simplified_biconditionals
            .into_iter()
            .flat_map(Clauses::from)
            .collect();
        let mut max: usize = 0;
        cnf.iter().for_each(|clause| {
            let clause_max = clause.iter().max().expect("Unexpected empty clause");
            if max < *clause_max as usize {
                max = *clause_max as usize;
            }
        });
        cnf.push(vec![max as isize]);
        cnf.into()
    }
}

fn nodes_to_literals(nodes: &[usize], node_literals: &[Literal]) -> Vec<Literal> {
    nodes.iter().map(|node| node_literals[*node]).collect()
}

fn simplify_biconditionals<'a>(
    ddnnf: &Ddnnf,
    biconditionals: &'a mut Vec<Biconditional>,
) -> Vec<&'a mut Biconditional> {
    let biconditionals = apply_core_dead_taut_decision_propagation(ddnnf, biconditionals);

    biconditionals
}

fn apply_core_dead_taut_decision_propagation<'a>(
    ddnnf: &Ddnnf,
    biconditionals: &'a mut Vec<Biconditional>,
) -> Vec<&'a mut Biconditional> {
    let backbone = ddnnf.get_core();
    let mut current_decisions: Vec<Literal> = Vec::new();
    let mut next_decisions: Vec<Literal> = Vec::new();
    let mut discarded_by_propagation: Vec<usize> = Vec::new(); // Remember tseitin variables that are not needed anymore

    for lit in backbone {
        current_decisions.push(lit as isize);
    }
    while !current_decisions.is_empty() {
        println!("{:#?}", current_decisions);
        biconditionals.iter_mut().for_each(|biconditional| {
            if !discarded_by_propagation.contains(&biconditional.index) {
                if biconditional.is_taut() {
                    discarded_by_propagation.push(biconditional.index);
                } else {
                    let decision = biconditional.decision_propagation(&current_decisions);
                    if decision != 0 {
                        next_decisions.push(decision);
                        discarded_by_propagation.push(biconditional.index);
                    }
                }
            }
        });
        current_decisions = next_decisions;
        next_decisions = Vec::new();
    }
    return cleanup_after_variable_elimination(biconditionals, &discarded_by_propagation);
}

fn cleanup_after_variable_elimination<'a>(
    biconditionals: &'a mut Vec<Biconditional>,
    eliminated_variables: &Vec<usize>,
) -> Vec<&'a mut Biconditional> {
    let mut resulting_biconditionals = Vec::new();
    let mut new_indices = Vec::new();

    let mut tseitin_starting_index: usize = usize::MAX;
        biconditionals.iter().for_each(|biconditional| {
            if tseitin_starting_index > biconditional.index {
                tseitin_starting_index = biconditional.index;
            }
    });

    biconditionals.iter_mut().for_each(|biconditional| {
        if !eliminated_variables.contains(&biconditional.index) {
            new_indices.push(biconditional.index.clone());
            resulting_biconditionals.push(biconditional);
        }
    });
    new_indices.sort();

    new_indices.iter().enumerate().for_each(|(index, value)| {
        resulting_biconditionals
            .iter_mut()
            .for_each(|biconditional| {
                biconditional.replace_variable_index(*value, index + tseitin_starting_index);
            });
    });

    return resulting_biconditionals;
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
