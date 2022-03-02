use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::{map_res, recognize, value},
    multi::many1,
    sequence::{pair, preceded},
    IResult,
};

use TokenIdentifier::{And, False, Header, NegativeLiteral, Or, PositiveLiteral, True};

#[derive(Debug, Copy, Clone, PartialEq)]
/// Every token gets an enum instance for the lexing progress
pub enum TokenIdentifier {
    /// The header of the nnf file
    Header,
    /// A inner node that contains atleast one child
    And,
    /// A inner node that contains exactly two child nodes
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

/// Combination of a `TId` and a vector that contains either the child nodes or the variable number of a literal
pub type Token = (TokenIdentifier, Vec<usize>);

/// Just an abbreviation for `TokenIdentifier`
pub type TId = TokenIdentifier;

/// Tests all parsers for a given input string and returns the result of the fitting parser.
/// It is important that we check if a line is a True or a False leaf before we check for an inner node,
/// because both start with the same latter but `lex_false` and `lex_true` are way more specific and therefore
/// just match in that edge case.
///
/// # Examples
///
/// ```
/// extern crate ddnnf_lib;
/// use ddnnf_lib::parser::lexer::*;
///
/// let and_str = "A 3 1 2 3";
///
/// // the output is Ok(("", (And, vec![3, 1, 2]))) but because of multiple missing
/// // PartialEq Traits we can not directly assert that for equality
/// assert_eq!(lex_line(and_str).unwrap().1.0, TId::And);
/// // note that the order of the childs have changed due to the usage of
/// // swap_remove() with O(1) instead of remove() with O(n)
/// assert_eq!(lex_line(and_str).unwrap().1.1, vec![1_usize, 2_usize, 3_usize]);
///
/// let header_str = "nnf 32 13 23";
/// assert_eq!(lex_line(header_str).unwrap().1.0, TId::Header);
/// assert_eq!(lex_line(header_str).unwrap().1.1, vec![32_usize, 13_usize, 23_usize]);
///
/// // same problem with PartialEq for the following
/// let or_str = "O 10 2 20 24";
/// assert_eq!(lex_line(or_str), Ok(("", (TId::Or, vec![10, 20, 24]))));
///
/// let negative_literal_str = "L -3";
/// assert_eq!(lex_line(negative_literal_str), Ok(("", (TId::NegativeLiteral, vec![3]))));
///
/// let true_str = "A 0";
/// assert_eq!(lex_line(true_str), Ok(("", (TId::True, vec![]))));
/// ```
#[inline]
pub fn lex_line(line: &str) -> IResult<&str, Token> {
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
fn lex_header(line: &str) -> IResult<&str, Token> {
    map_res(
        preceded(tag("nnf"), parse_alt_space1_number1),
        get_number1_wrapper(Header),
    )(line)
}

// Lexes a And node which is a inner node with the format "A c i1 i2 ... ic" where c is the number
// of childs the node has and i1, i2, ... are the references to the actual child nodes.
fn lex_and(line: &str) -> IResult<&str, Token> {
    map_res(
        preceded(char('A'), parse_alt_space1_number1),
        get_number1_wrapper(And),
    )(line)
}

// Lexes a Or node which is a inner node with the format "O j c i1 i2" with i1 and i2 as childs.
fn lex_or(line: &str) -> IResult<&str, Token> {
    map_res(
        preceded(char('O'), parse_alt_space1_number1),
        get_number1_wrapper(Or),
    )(line)
}

// Lexes a positive Literal node which is a leaf node with the format "L i1" with i1 as the variable number.
fn lex_positive_literal(line: &str) -> IResult<&str, Token> {
    map_res(
        preceded(tag("L "), recognize(digit1)),
        get_number_wrapper(PositiveLiteral),
    )(line)
}

// Lexes a negative Literal node which is a leaf node with the format "L -i1" with i1 as the variable number.
fn lex_negative_literal(line: &str) -> IResult<&str, Token> {
    map_res(
        preceded(tag("L "), preceded(char('-'), digit1)),
        get_number_wrapper(NegativeLiteral),
    )(line)
}

// Lexes a True node which is a leaf node with the format "A 0".
fn lex_true(line: &str) -> IResult<&str, Token> {
    value((True, Vec::new()), tag("A 0"))(line)
}

// Lexes a False node which is a leaf node with the format "O 0 0".
fn lex_false(line: &str) -> IResult<&str, Token> {
    value((False, Vec::new()), tag("O 0 0"))(line)
}

// Returns a closure that lexes an alternating sequence of spaces and multiple digits which form a number.
// Minus signs which indicate a negative number are not rational for our use case because we want usizes.
//
// Panics for kinds which are not one of the following: Header, And, Or because other TokenIdentifier
// do not hold childs.
fn get_number1_wrapper(kind: TId) -> Box<dyn Fn(&str) -> Result<Token, &str>> {
    Box::new(move |s: &str| -> Result<Token, &str> {
        let mut r: Vec<usize> = Vec::new();
        for v in s.split_whitespace() {
            r.push(v.parse::<usize>().unwrap());
        }

        // remove the unnecessary numbers of And and Or
        match (kind, r) {
            (And, mut r) => {
                r.remove(0); //swap_remove() has O(1) because the empty space gets filled with the last element
                Ok((And, r))
            }
            (Or, mut r) => {
                r.remove(1);
                Ok((Or, r))
            }
            (Header, r) => Ok((Header, r)),
            _ => panic!(
                "Tried to parse an alternating sequence after NodeType {:?}",
                kind
            ),
        }
    })
}

// Returns a closure that lexes exactly one number which consists of multiple digits to form an usize
fn get_number_wrapper(kind: TId) -> Box<dyn Fn(&str) -> Result<Token, &str>> {
    Box::new(move |s: &str| -> Result<Token, &str> {
        Ok((kind, vec![s.parse::<usize>().unwrap()]))
    })
}

// parses an alternating sequence of one space and multiple digits which form a number or to be
// specific a usize.
fn parse_alt_space1_number1(input: &str) -> IResult<&str, &str> {
    recognize(many1(pair(char(' '), digit1)))(input)
}

pub fn serialize_token(token: Token) -> String {
    match token {
        (Header, v) => format!("nnf {} {} {}\n", v[0], v[1], v[2]),
        (And, v) => {
            let mut s = String::from("A ");
            s.push_str(&v.len().to_string());
            s.push(' ');

            for n in 0..v.len() {
                s.push_str(&v[n].to_string());

                if n != v.len() - 1 {
                    s.push(' ');
                } else {
                    s.push('\n');
                }
            }
            s
        }
        (Or, v) => format!("O {} 2 {} {}\n", v[0], v[1], v[2]),
        (PositiveLiteral, v) => format!("L {}\n", v[0]),
        (NegativeLiteral, v) => format!("L -{}\n", v[0]),
        (False, _) => String::from("O 0 0\n"),
        (True, _) => String::from("A 0\n"),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_lex_lines() {
        let and_str = "A 3 1 2 3";
        let or_str = "O 10 2 40 44";
        let positive_literal_str = "L 3";

        assert_eq!(lex_line(and_str).unwrap().1 .0, And);
        assert_eq!(
            lex_line(and_str).unwrap().1 .1,
            vec![1_usize, 2_usize, 3_usize]
        );

        assert_eq!(lex_line(or_str).unwrap().1 .0, Or);
        assert_eq!(
            lex_line(or_str).unwrap().1 .1,
            vec![10_usize, 40_usize, 44_usize]
        );

        assert_eq!(
            lex_line(positive_literal_str).unwrap().1 .0,
            PositiveLiteral
        );
        assert_eq!(lex_line(positive_literal_str).unwrap().1 .1, vec![3_usize]);
    }

    #[test]
    fn test_individual_lexer() {
        let header_str = "nnf 32 13 23";
        assert_eq!(lex_header(header_str).unwrap().1 .0, Header);
        assert_eq!(
            lex_header(header_str).unwrap().1 .1,
            vec![32_usize, 13_usize, 23_usize]
        );

        let and_str = "A 1 1";
        assert_eq!(lex_and(and_str).unwrap().1 .0, And);
        assert_eq!(lex_and(and_str).unwrap().1 .1, vec![1_usize]);

        let or_str = "O 10 2 20 24";
        assert_eq!(lex_or(or_str).unwrap().1 .0, Or);
        assert_eq!(
            lex_or(or_str).unwrap().1 .1,
            vec![10_usize, 20_usize, 24_usize]
        );

        let positive_literal_str = "L 13";
        assert_eq!(
            lex_positive_literal(positive_literal_str).unwrap().1 .0,
            PositiveLiteral
        );
        assert_eq!(
            lex_positive_literal(positive_literal_str).unwrap().1 .1,
            vec![13_usize]
        );

        let negative_literal_str = "L -113";
        assert_eq!(
            lex_negative_literal(negative_literal_str).unwrap().1 .0,
            NegativeLiteral
        );
        assert_eq!(
            lex_negative_literal(negative_literal_str).unwrap().1 .1,
            vec![113_usize]
        );

        let v: Vec<usize> = Vec::new();

        let true_str = "A 0";
        assert_eq!(lex_true(true_str).unwrap().1 .0, True);
        assert_eq!(lex_true(true_str).unwrap().1 .1, v.clone());

        let false_str = "O 0 0";
        assert_eq!(lex_false(false_str).unwrap().1 .0, False);
        assert_eq!(lex_false(false_str).unwrap().1 .1, v);
    }

    #[test]
    fn test_serialization() {
        let header: Token = (Header, vec![10, 20, 30]);
        let header_s: String = String::from("nnf 10 20 30\n");
        assert_eq!(serialize_token(header), header_s);

        let and: Token = (And, vec![1, 2, 3, 4, 5]);
        let and_s: String = String::from("A 5 1 2 3 4 5\n");
        assert_eq!(serialize_token(and), and_s);

        let or: Token = (Or, vec![42, 1, 2]);
        let or_s: String = String::from("O 42 2 1 2\n");
        assert_eq!(serialize_token(or), or_s);

        let p_literal: Token = (PositiveLiteral, vec![4269]);
        let p_literal_s: String = String::from("L 4269\n");
        assert_eq!(serialize_token(p_literal), p_literal_s);

        let n_literal: Token = (NegativeLiteral, vec![4269]);
        let n_literal_s: String = String::from("L -4269\n");
        assert_eq!(serialize_token(n_literal), n_literal_s);

        let ttrue: Token = (True, vec![]);
        let ttrue_s: String = String::from("A 0\n");
        assert_eq!(serialize_token(ttrue), ttrue_s);

        let tfalse: Token = (False, vec![]);
        let tfalse_s: String = String::from("O 0 0\n");
        assert_eq!(serialize_token(tfalse), tfalse_s);
    }
}
