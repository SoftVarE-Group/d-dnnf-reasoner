use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1, space1},
    combinator::{map, recognize, value},
    multi::many_m_n,
    sequence::{pair, preceded, terminated},
    IResult,
};

use D4Token::*;

#[derive(Debug, Clone, PartialEq)]
/// Every token gets an enum instance for the D4lexing progress
pub enum D4Token {
    /// An inner node that contains atleast one child
    And,
    /// An inner node that contains atleast one child
    Or,
    /// A True node which is the sink of of the DAG
    True,
    /// A False node can exist, but is rather an exception than the norm
    False,
    /// An Edge between two nodes, with
    Edge {
        from: i32,
        to: i32,
        features: Vec<i32>,
    },
}

/// Tests all parsers for a given input string and returns the result of the fitting parser.
/// We sort the alternatives by their absolute occurences in a ddnnf, starting from most occuring
/// to least occuring, to lex faster
#[inline]
pub fn lex_line_d4(line: &str) -> IResult<&str, D4Token> {
    alt((lex_edge, lex_or, lex_and, lex_true, lex_false))(line)
}

// Lexes an Edge Node with the format "F T F1 F2 F3 ... 0" with F being the from and T being the to node of the edge.
// Further, F1, F2, F3, ... are the features
fn lex_edge(line: &str) -> IResult<&str, D4Token> {
    map(
        terminated(
            recognize(many_m_n(2, usize::MAX, parse_signed_alt_space1_number1)),
            tag("0"),
        ),
        |out: &str| {
            let ws_numbers: Vec<i32> = out
                .split_whitespace()
                .map(|num: &str| {
                    num.parse::<i32>().unwrap_or_else(|_| {
                        panic!("Was not able to parse i32 for edge. String was {}", out)
                    })
                })
                .collect::<Vec<i32>>();
            Edge {
                from: ws_numbers[0],
                to: ws_numbers[1],
                features: ws_numbers[2..ws_numbers.len()].to_vec(),
            }
        },
    )(line)
}

// lexes multiple sequences of signed numbers
pub(super) fn parse_signed_alt_space1_number1(line: &str) -> IResult<&str, (&str, &str)> {
    alt((pair(digit1, space1), pair(neg_digit1, space1)))(line)
}

// lexes a sequence of numbers that start with a minues sign
fn neg_digit1(line: &str) -> IResult<&str, &str> {
    recognize(pair(char('-'), digit1))(line)
}

// Lexes an And node which is a inner node with the format "a N 0" with N as Node number.
fn lex_and(line: &str) -> IResult<&str, D4Token> {
    value(And, preceded(tag("a "), digit1))(line)
}

// Lexes an Or node which is a inner node with the format "o N 0" with N as Node number.
fn lex_or(line: &str) -> IResult<&str, D4Token> {
    value(Or, preceded(tag("o "), digit1))(line)
}

// Lexes a True node which is a leaf node with the format "t N 0" with N as Node number.
fn lex_true(line: &str) -> IResult<&str, D4Token> {
    value(True, preceded(tag("t "), digit1))(line)
}

// Lexes a False node which is a leaf node with the format "f N 0"  with N as Node number.
fn lex_false(line: &str) -> IResult<&str, D4Token> {
    value(False, preceded(tag("f "), digit1))(line)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn individual_lexer_d4() {
        let and_str = "a 1 0";
        let or_str = "o 2 0";
        let true_str = "t 3 0";
        let false_str = "f 4 0";
        let edge_str = "2 3 4 -5 0";
        let failed_edge = "2 3 -5 THREE 0";

        assert_eq!(lex_and(and_str).unwrap().1, And);
        assert_eq!(lex_line_d4(and_str).unwrap().1, And,);

        assert_eq!(lex_or(or_str).unwrap().1, Or);
        assert_eq!(lex_line_d4(or_str).unwrap().1, Or);

        assert_eq!(lex_true(true_str).unwrap().1, True,);

        assert_eq!(lex_line_d4(true_str).unwrap().1, True,);

        assert_eq!(lex_false(false_str).unwrap().1, False,);

        assert_eq!(lex_line_d4(false_str).unwrap().1, False,);

        assert_eq!(
            lex_edge(edge_str).unwrap().1,
            Edge {
                from: 2,
                to: 3,
                features: vec![4, -5]
            },
        );

        assert_eq!(
            lex_line_d4(edge_str).unwrap().1,
            Edge {
                from: 2,
                to: 3,
                features: vec![4, -5]
            },
        );

        let result = std::panic::catch_unwind(|| lex_edge(failed_edge)).unwrap();
        assert!(result.is_err());

        let result = std::panic::catch_unwind(|| lex_line_d4(failed_edge)).unwrap();
        assert!(result.is_err());
    }
}
