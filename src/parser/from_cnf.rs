//! A lexer that categorizes a CNF into its coresponding tokens.

use std::{io::{BufReader, BufRead, Write}, fs::File, cmp::max, collections::HashSet};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, value},
    sequence::preceded,
    IResult,
};

use crate::{c2d_lexer::{parse_alt_space1_number1, split_numbers}, d4_lexer::parse_signed_alt_space1_number1};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A classification for the different kinds of lines a CNF file contains
pub enum CNFToken {
    /// A comment in the CNF. It starts with a 'c '
    Comment,
    /// A clause that consists of a sequence of (signed) numbers
    Clause,
    /// The head of a CNF file of the format p cnf #FEATURES #CLAUSESs
    Header {
        /// The number of features in the CNF
        features: usize,
        /// The number of clauses in the CNF
        clauses: usize,
    },
}

use CNFToken::*;

use super::util::format_vec;

/// Lexes a line and checks whether it is a CNF header, comment or clause.
/// We are only interested in the header, because it contains the information about the number of features.
#[inline]
pub fn check_for_cnf_header(line: &str) -> IResult<&str, CNFToken> {
    alt((
        lex_comment,
        lex_header,
        lex_clause,
    ))(line)
}

// lexes the head of a CNF file of the format p cnf #FEATURES #CLAUSES
fn lex_header(line: &str) -> IResult<&str, CNFToken> {
    map(
        preceded(tag("p cnf"), parse_alt_space1_number1),
        |out: &str| {
            let nums: Vec<usize> = split_numbers(out);
            Header { features: nums[0], clauses: nums[1] }
        },
    )(line)
}

// identifies a CNF clause by finding a sequence of (signed) numbers
fn lex_clause(line: &str) -> IResult<&str, CNFToken> {
    value(Clause, parse_signed_alt_space1_number1)(line)
}

// identifies a CNF comment by its leading 'c'
fn lex_comment(line: &str) -> IResult<&str, CNFToken> {
    value(Comment, char('c'))(line)
}

/// Appends the given clause to the CNF file. This function also updates the Header by incrementing
/// the number of clauses as well as the number of variables if necessary.
pub fn add_clause_cnf(path: &str, clause: &[i32]) {
    let file = File::open(path).unwrap();
    let lines = BufReader::new(file).lines();
    let mut manipulated_cnf = Vec::new();
    
    for line in lines {
        let line = line.expect("Unable to read line");
        match check_for_cnf_header(line.as_str()).unwrap().1 {
            Header { features, clauses } => {
                let max_feature_number = max(features, clause.iter().map(|f| f.unsigned_abs()).max().unwrap() as usize);
                manipulated_cnf.push(format!("p cnf {max_feature_number} {}", clauses + 1))
            },
            Comment | Clause => manipulated_cnf.push(line),
        }
    }
    manipulated_cnf.push(format!("{} 0", format_vec(clause.iter())));

    let mut wfile = File::create(path).unwrap();
    wfile.write_all(manipulated_cnf.join("\n").as_bytes()).unwrap();
}

/// Removes the last amount many clauses from the CNF. The header gets also updated.
/// The number of clauses automatically, the total number of features if supplied.
pub fn remove_tail_clauses_cnf(path: &str, total_features: Option<usize>, amount: usize) {
    let mut file = File::open(path).unwrap();
    let lines = BufReader::new(file).lines();
    let mut manipulated_cnf = Vec::new();
    
    for line in lines {
        let line = line.expect("Unable to read line");
        match check_for_cnf_header(line.as_str()).unwrap().1 {
            Header { features, clauses } => {
                manipulated_cnf.push(format!("p cnf {} {}", total_features.unwrap_or(features), clauses - amount))
            },
            Comment | Clause => manipulated_cnf.push(line),
        }
    }
    manipulated_cnf.truncate(manipulated_cnf.len() - amount);

    file = File::create(path).unwrap();
    file.write_all(manipulated_cnf.join("\n").as_bytes()).unwrap();
}

/// Removes a specific clause from the CNF. Also removes duplicates. Does update the Header accordingly.
pub fn remove_clause_cnf(path: &str, clause: &[i32], total_features: Option<usize>) {
    let mut file = File::open(path).unwrap();
    let lines = BufReader::new(file).lines();
    let mut manipulated_cnf = Vec::new();
    
    for line in lines {
        let line = line.expect("Unable to read line");
        match check_for_cnf_header(line.as_str()).unwrap().1 {
            Header { features, clauses } => {
                manipulated_cnf.push(format!("p cnf {} {}", total_features.unwrap_or(features), clauses - 1))
            },
            Comment | Clause => {
                if line != format!("{} 0", format_vec(clause.iter())) { // ignore clause
                    manipulated_cnf.push(line)
                }
            },
        }
    }

    file = File::create(path).unwrap();
    file.write_all(manipulated_cnf.join("\n").as_bytes()).unwrap();
}

/// Reads a CNF file and returns all the contained clauses.
pub fn get_all_clauses_cnf(path: &str) -> Vec<Vec<i32>> {
    let file = File::open(path).unwrap();
    let lines = BufReader::new(file).lines();
    let mut clauses = Vec::new();
    
    for line in lines {
        let line = line.expect("Unable to read line");
        match check_for_cnf_header(line.as_str()).unwrap().1 {
            Clause => {
                let mut clause: Vec<i32> = line.split(" ").map(|num| num.parse::<i32>().unwrap()).collect();
                clause.remove(clause.len() - 1); // remove the trailing 0
                clauses.push(clause);
            },
            _ => ()
        }
    }
    clauses
}

/// Simplifys the clauses according to the following rules
/// 1) If a variable occurs multiple times -> remove all occurences except one
/// 2) If a variable occurs singed and unsinged (i.e. [a , b, -a]) -> remove whole clause
pub fn simplify_clauses(mut clauses: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut clause_set: Vec<HashSet<i32>> = Vec::with_capacity(clauses.len());

    // Remove duplicates as specified in 1)
    for inner_vec in clauses.iter_mut() {
        clause_set.push(inner_vec.drain(..).collect());
    }

    // Remove clauses that are always SAT as in 2)
    clause_set.retain(|clause| clause.iter().all(|elem| !clause.contains(&-elem)));

    let mut decisions = HashSet::new();
    clause_set.iter().for_each(|clause| 
        if clause.len() == 1 {
            decisions.insert(clause.clone().into_iter().collect::<Vec<_>>()[0]);
        }
    );

    let mut simplified_clauses = Vec::with_capacity(clause_set.len());
    for clause in clause_set.into_iter() {
        simplified_clauses.push(clause.into_iter().collect());
    }

    (simplified_clauses, decisions) = apply_decisions(simplified_clauses, decisions);
    decisions.into_iter().for_each(|decision| simplified_clauses.push(vec![decision]));

    simplified_clauses
}

/// Repeatedly applys decisions. After the inital set of decisions is applied, clauses might
/// be reduced to a point where they also become a decision, til there are no new decisions to be applied.
pub fn apply_decisions(mut relevant_clauses: Vec<Vec<i32>>, mut accumlated_decisions: HashSet<i32>) -> (Vec<Vec<i32>>, HashSet<i32>) {
    let mut decisions = accumlated_decisions.clone();
    while !decisions.is_empty() {
        let mut new_decisions = HashSet::new();
        let mut reduced_clauses = Vec::new();

        for mut clause in relevant_clauses {
            // clause is not already satisfied in other parts of the dDNNF
            if !clause.iter().any(|variable| decisions.contains(variable)) {
                // remove variable assignments that aren't valid anymore
                clause.retain(|variable| !decisions.contains(&-variable));
                if !clause.is_empty() {
                    if clause.len() == 1 {
                        new_decisions.insert(clause[0]);
                    }
                    reduced_clauses.push(clause);
                }
            }
        }

        accumlated_decisions.extend(decisions);
        relevant_clauses = reduced_clauses;
        decisions = new_decisions;
    }

    (relevant_clauses, accumlated_decisions)
}

/// Performs the same simplifications as [simplify_clauses],
/// but only on one clause.
/// Returns None if the clause is UNSAT and Some(empty vector) if the clause is SAT.
pub fn reduce_clause(clause: &[i32], decisions: &HashSet<i32>) -> Option<Vec<i32>> {
    if clause.is_empty() { return Some(vec![]) }
    let mut set_clause: HashSet<i32> = HashSet::new();
    for elem in clause {
        if set_clause.contains(&-elem) || decisions.contains(elem) {
            return Some(vec![]);
        } else if !decisions.contains(&-elem) {
            set_clause.insert(*elem);
        }
    }

    if set_clause.is_empty() { return None; }

    Some(set_clause.into_iter().collect())
}

#[cfg(test)]
mod test {
    use std::fs::{self, read_to_string};

    use super::*;

    #[test]
    fn lex_cnf_lines() {
        let comment = "c 1 N_100300__F_100332";
        let header  = "p cnf 2513 10275";
        let clause = "-1628 1734 0";

        assert_eq!(check_for_cnf_header(comment).unwrap().1, Comment);
        assert_eq!(check_for_cnf_header(header).unwrap().1, Header { features: 2513, clauses: 10275 } );
        assert_eq!(check_for_cnf_header(clause).unwrap().1, Clause );
    }

    #[test]
    fn manipulate_cnf() {
        const INTER_CNF_PATH: &str = ".inter.cnf";

        let cnf = "p cnf 3 3\n1 2 0\n-1 -2 0\n1 2 3 0";
        let mut file = File::create(INTER_CNF_PATH).unwrap();
        file.write_all(cnf.as_bytes()).unwrap();

        add_clause_cnf(INTER_CNF_PATH, &vec![3, -3]);
        assert_eq!(
            "p cnf 3 4\n1 2 0\n-1 -2 0\n1 2 3 0\n3 -3 0",
            read_to_string(INTER_CNF_PATH).unwrap()
        );

        add_clause_cnf(INTER_CNF_PATH, &vec![1, 2, 3, 4]);
        assert_eq!(
            "p cnf 4 5\n1 2 0\n-1 -2 0\n1 2 3 0\n3 -3 0\n1 2 3 4 0",
            read_to_string(INTER_CNF_PATH).unwrap()
        );

        remove_tail_clauses_cnf(INTER_CNF_PATH, None, 1);
        assert_eq!(
            "p cnf 4 4\n1 2 0\n-1 -2 0\n1 2 3 0\n3 -3 0",
            read_to_string(INTER_CNF_PATH).unwrap()
        );

        remove_tail_clauses_cnf(INTER_CNF_PATH, Some(3), 2);
        assert_eq!(
            "p cnf 3 2\n1 2 0\n-1 -2 0",
            read_to_string(INTER_CNF_PATH).unwrap()
        );

        add_clause_cnf(INTER_CNF_PATH, &vec![3, -3]);
        remove_clause_cnf(INTER_CNF_PATH, &vec![-1, -2], None);
        assert_eq!(
            "p cnf 3 2\n1 2 0\n3 -3 0",
            read_to_string(INTER_CNF_PATH).unwrap()
        );

        assert_eq!(
            vec![vec![1, 2], vec![3, -3]],
            get_all_clauses_cnf(INTER_CNF_PATH)
        );

        fs::remove_file(INTER_CNF_PATH).unwrap();
    }

    #[test]
    fn clause_simplfication() {
        let clauses = vec![
            vec![1, 2, 3],
            vec![1, 2, 1, 1, 2, 3, 2, 2, 9, 1, 1, 1, 1, 3, 3, 1, 2],
            vec![1, 1, 2, 2, 3, 2, 2, 2, 5, -1, 8, 9]
        ];

        let mut simplified_clauses = simplify_clauses(clauses);
        for clause in simplified_clauses.iter_mut() {
            clause.sort();
        }

        assert_eq!(
            vec![
                vec![1, 2, 3],
                vec![1, 2, 3, 9],
            ],
            simplified_clauses
        )
    }

    #[test]
    fn clause_reduction() {
        let contains_same_elements = |fst: Option<Vec<i32>>, snd: Option<Vec<i32>>| {
            if fst.is_none() || snd.is_none() {
                return fst == snd;
            }

            let fst = fst.unwrap(); let snd = snd.unwrap(); 
            
            fst.len() == snd.len()
            && fst.iter().all(|elem| snd.contains(elem))
            && snd.iter().all(|elem| fst.contains(elem))
        };

        assert!(contains_same_elements(
            Some(vec![1, 2, 3]),
            reduce_clause(&vec![1, 2, 3, 3, 3, 2], &HashSet::new())
        ));

        assert!(contains_same_elements(
            Some(vec![]),
            reduce_clause(&vec![], &vec![1, 2].into_iter().collect())
        ));

        assert!(contains_same_elements(
            Some(vec![]),
            reduce_clause(&vec![1, 2, 3], &vec![-1, 2].into_iter().collect())
        ));

        assert!(contains_same_elements(
            Some(vec![3]),
            reduce_clause(&vec![1, 2, 3], &vec![-1, -2].into_iter().collect())
        ));

        assert!(contains_same_elements(
            None,
            reduce_clause(&vec![1, 2, 3], &vec![-1, -2, -3].into_iter().collect())
        ));
    }

    #[test]
    fn clauses_apply_decisions() {
        let mut clauses = vec![
            vec![1, 2, 3],
            vec![-2, 3],
            vec![2],
            vec![5, 3],
            vec![-1, -2, -3, 4]
        ];
        let mut decisions = vec![1].into_iter().collect();

        (clauses, decisions) = apply_decisions(clauses, decisions);

        assert_eq!(vec![1, 2, 3, 4].into_iter().collect::<HashSet<_>>(), decisions);
        assert_eq!(Vec::<Vec<i32>>::new(), clauses);
    }
}