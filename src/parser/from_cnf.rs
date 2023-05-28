use nom::{
    branch::alt,
    bytes::complete::tag,
    character::{complete::char},
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
        features: usize,
        clauses: usize,
    },
}

use CNFToken::*;

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

#[cfg(test)]
mod test {
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
}