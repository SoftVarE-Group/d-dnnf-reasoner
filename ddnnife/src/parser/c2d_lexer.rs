use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::{map, recognize, value},
    multi::many1,
    sequence::{pair, preceded},
    IResult,
};

#[derive(Copy, Clone, PartialEq)]
/// Every token gets an enum instance for the lexing progress
pub enum TokenIdentifier {
    /// The header of the nnf file
    Header,
    /// An inner node that contains atleast one child
    And,
    /// An inner node that contains exactly two child nodes
    Or,
    /// A leaf node that countains a positive number of a variable
    PositiveLiteral,
    /// A leaf node that countains a negated number of a variable
    NegativeLiteral,
    /// A special And node that has zero childs
    True,
    /// A special Or node that has zero childs
    False,
}

/// Just an abbreviation for `TokenIdentifier`
pub type TId = TokenIdentifier;

use C2DToken::*;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Every C2DToken gets an enum instance for the lexing progress
pub enum C2DToken {
    /// The header of the nnf file
    Header {
        nodes: usize,
        edges: usize,
        variables: usize,
    },
    /// An inner node that contains atleast one child
    And { children: Vec<usize> },
    /// An inner node that contains exactly two child nodes
    Or { children: Vec<usize>, decision: u32 },
    /// A leaf node that countains a positive/negative number of a variable
    Literal { feature: i32 },
    /// A special And node that has zero childs
    True,
    /// A special Or node that has zero childs
    False,
}

/// Tests all parsers for a given input string and returns the result of the fitting parser.
/// It is important that we check if a line is a True or a False leaf before we check for an inner node,
/// because both start with the same latter but `lex_false` and `lex_true` are way more specific and therefore
/// just match in that edge case.
///
/// # Examples
///
/// ```
/// use ddnnife::parser::c2d_lexer::*;
///
/// let and_str = "A 3 1 2 3";
/// assert_eq!(lex_line_c2d(and_str).unwrap().1, C2DToken::And { children: vec![1_usize, 2_usize, 3_usize]});
///
/// let header_str = "nnf 32 13 23";
/// assert_eq!(lex_line_c2d(header_str).unwrap().1, C2DToken::Header { nodes: 32_usize, edges: 13_usize, variables: 23_usize});
///
/// // same problem with PartialEq for the following
/// let or_str = "O 10 2 20 24";
/// assert_eq!(lex_line_c2d(or_str), Ok(("", C2DToken::Or { decision: 10, children: vec![20, 24] } )));
///
/// let negative_literal_str = "L -3";
/// assert_eq!(lex_line_c2d(negative_literal_str), Ok(("", C2DToken::Literal { feature: -3 } )));
///
/// let true_str = "A 0";
/// assert_eq!(lex_line_c2d(true_str), Ok(("", C2DToken::True)));
/// ```
#[inline]
pub fn lex_line_c2d(line: &str) -> IResult<&str, C2DToken> {
    alt((
        lex_header,
        lex_true,
        lex_false,
        lex_and,
        lex_or,
        lex_positive_literal,
        lex_negative_literal,
    ))(line)
}

// Lexes the header and only the header with the format "nnf v e n" where v is the number of nodes in the NNF,
// e is the number of edges in the NNF, and n is the number of variables over which the NNF is defined.
fn lex_header(line: &str) -> IResult<&str, C2DToken> {
    map(
        preceded(tag("nnf"), parse_alt_space1_number1),
        |out: &str| {
            let nums: Vec<usize> = split_numbers(out);
            Header {
                nodes: nums[0],
                edges: nums[1],
                variables: nums[2],
            }
        },
    )(line)
}

// Lexes a And node which is a inner node with the format "A c i1 i2 ... ic" where c is the number
// of childs the node has and i1, i2, ... are the references to the actual child nodes.
fn lex_and(line: &str) -> IResult<&str, C2DToken> {
    map(
        preceded(char('A'), parse_alt_space1_number1),
        |out: &str| {
            let mut nums: Vec<usize> = split_numbers(out);
            nums.remove(0); // remove information about number of children
            And { children: nums }
        },
    )(line)
}

// Lexes a Or node which is a inner node with the format "O j c i1 i2" with i1 and i2 as childs.
fn lex_or(line: &str) -> IResult<&str, C2DToken> {
    map(
        preceded(char('O'), parse_alt_space1_number1),
        |out: &str| {
            let mut nums: Vec<usize> = split_numbers(out);
            let dec = nums.remove(0) as u32;
            nums.remove(0); // remove information about number of children
            Or {
                decision: dec,
                children: nums,
            }
        },
    )(line)
}

// Lexes a positive Literal node which is a leaf node with the format "L i1" with i1 as the variable number.
fn lex_positive_literal(line: &str) -> IResult<&str, C2DToken> {
    map(preceded(tag("L "), recognize(digit1)), |s: &str| Literal {
        feature: s.parse::<i32>().unwrap(),
    })(line)
}

// Lexes a negative Literal node which is a leaf node with the format "L i1" with i1 as the variable number.
fn lex_negative_literal(line: &str) -> IResult<&str, C2DToken> {
    map(
        preceded(tag("L "), recognize(pair(char('-'), digit1))),
        |s: &str| Literal {
            feature: s.parse::<i32>().unwrap(),
        },
    )(line)
}

// Lexes a True node which is a leaf node with the format "A 0".
fn lex_true(line: &str) -> IResult<&str, C2DToken> {
    value(True, tag("A 0"))(line)
}

// Lexes a False node which is a leaf node with the format "O 0 0".
fn lex_false(line: &str) -> IResult<&str, C2DToken> {
    value(False, tag("O 0 0"))(line)
}

// Returns a closure that lexes exactly one number which consists of multiple digits to form an T
pub(super) fn split_numbers<T: std::str::FromStr>(out: &str) -> Vec<T> {
    out.split_whitespace()
        .map(|num: &str| {
            num.parse::<T>().unwrap_or_else(|_| {
                panic!(
                    "Was not able to parse {} for node. String was {}",
                    std::any::type_name::<T>(),
                    out
                )
            })
        })
        .collect::<Vec<T>>()
}

// parses an alternating sequence of one space and multiple digits which form a number
pub(super) fn parse_alt_space1_number1(input: &str) -> IResult<&str, &str> {
    recognize(many1(pair(char(' '), digit1)))(input)
}

#[allow(non_snake_case)]
pub fn deconstruct_C2DToken(C2DToken: C2DToken) -> String {
    match C2DToken {
        Header {
            nodes,
            edges,
            variables,
        } => format!("nnf {} {} {}\n", nodes, edges, variables),
        And { children: c } => {
            let mut s = String::from("A ");
            s.push_str(&c.len().to_string());
            s.push(' ');

            for n in 0..c.len() {
                s.push_str(&c[n].to_string());

                if n != c.len() - 1 {
                    s.push(' ');
                } else {
                    s.push('\n');
                }
            }
            s
        }
        Or {
            decision,
            children: c,
        } => format!("O {} 2 {} {}\n", decision, c[0], c[1]),
        Literal { feature } => format!("L {}\n", feature),
        False => String::from("O 0 0\n"),
        True => String::from("A 0\n"),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lex_lines() {
        let and_str = "A 3 11 12 13";
        let or_str = "O 10 2 40 44";
        let positive_literal_str = "L 3";
        let failed_and_str = "A THREE 11 TWELVE 13";

        assert_eq!(
            lex_line_c2d(and_str).unwrap().1,
            And {
                children: vec![11, 12, 13]
            }
        );

        assert_eq!(
            lex_line_c2d(or_str).unwrap().1,
            Or {
                decision: 10,
                children: vec![40, 44]
            }
        );

        assert_eq!(
            lex_line_c2d(positive_literal_str).unwrap().1,
            Literal { feature: 3 }
        );

        let result = std::panic::catch_unwind(|| lex_line_c2d(failed_and_str)).unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn individual_lexer() {
        let header_str = "nnf 32 13 23";
        assert_eq!(
            lex_header(header_str).unwrap().1,
            Header {
                nodes: 32,
                edges: 13,
                variables: 23
            }
        );

        let and_str = "A 1 1";
        assert_eq!(lex_and(and_str).unwrap().1, And { children: vec![1] });

        let or_str = "O 10 2 20 24";
        assert_eq!(
            lex_or(or_str).unwrap().1,
            Or {
                decision: 10,
                children: vec![20, 24]
            }
        );

        let positive_literal_str = "L 13";
        assert_eq!(
            lex_positive_literal(positive_literal_str).unwrap().1,
            Literal { feature: 13 }
        );

        let negative_literal_str = "L -113";
        assert_eq!(
            lex_negative_literal(negative_literal_str).unwrap().1,
            Literal { feature: -113 }
        );

        let true_str = "A 0";
        assert_eq!(lex_true(true_str).unwrap().1, True);

        let false_str = "O 0 0";
        assert_eq!(lex_false(false_str).unwrap().1, False);
    }

    #[test]
    fn serialization() {
        let header: C2DToken = Header {
            nodes: 10,
            edges: 20,
            variables: 30,
        };
        let header_s: String = String::from("nnf 10 20 30\n");
        assert_eq!(deconstruct_C2DToken(header), header_s);

        let and: C2DToken = And {
            children: vec![1, 2, 3, 4, 5],
        };
        let and_s: String = String::from("A 5 1 2 3 4 5\n");
        assert_eq!(deconstruct_C2DToken(and), and_s);

        let or: C2DToken = Or {
            decision: 42,
            children: vec![1, 2],
        };
        let or_s: String = String::from("O 42 2 1 2\n");
        assert_eq!(deconstruct_C2DToken(or), or_s);

        let p_literal: C2DToken = Literal { feature: 4269 };
        let p_literal_s: String = String::from("L 4269\n");
        assert_eq!(deconstruct_C2DToken(p_literal), p_literal_s);

        let n_literal: C2DToken = Literal { feature: -4269 };
        let n_literal_s: String = String::from("L -4269\n");
        assert_eq!(deconstruct_C2DToken(n_literal), n_literal_s);

        let ttrue: C2DToken = True;
        let ttrue_s: String = String::from("A 0\n");
        assert_eq!(deconstruct_C2DToken(ttrue), ttrue_s);

        let tfalse: C2DToken = False;
        let tfalse_s: String = String::from("O 0 0\n");
        assert_eq!(deconstruct_C2DToken(tfalse), tfalse_s);
    }
}
