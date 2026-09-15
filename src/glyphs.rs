//! An original 3x5 bitmap font, drawn to match the game's blocky shapes.
//!
//! Each glyph is five rows; the low three bits of each row are its pixels, with
//! bit 2 on the left. The font exists so titles and scores need no font asset.

/// Width of one glyph, in font pixels.
pub const GLYPH_WIDTH: usize = 3;
/// Height of one glyph, in font pixels.
pub const GLYPH_HEIGHT: usize = 5;
/// Blank columns between two adjacent glyphs.
const GLYPH_GAP: usize = 1;

/// The pixel rows for one character, or `None` if the font has no such glyph.
pub const fn glyph(character: char) -> Option<[u8; GLYPH_HEIGHT]> {
    Some(match character.to_ascii_uppercase() {
        ' ' => [0b000, 0b000, 0b000, 0b000, 0b000],
        'A' => [0b111, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b111, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b111, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b101, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b111],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' | '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b111, 0b101, 0b111, 0b100, 0b100],
        'Q' => [0b111, 0b101, 0b101, 0b111, 0b001],
        'R' => [0b111, 0b101, 0b110, 0b101, 0b101],
        'S' | '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b111, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b011, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b001, 0b001],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        _ => return None,
    })
}

/// Width of a rendered string in font pixels, gaps included.
pub fn text_width(text: &str) -> usize {
    let count = text.chars().count();
    if count == 0 {
        return 0;
    }
    count
        .saturating_mul(GLYPH_WIDTH)
        .saturating_add(count.saturating_sub(1).saturating_mul(GLYPH_GAP))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_is_five_rows_of_three_bits() {
        for character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars() {
            let found = glyph(character).expect("character has a glyph");
            assert_eq!(found.len(), GLYPH_HEIGHT);
            for row in found {
                assert!(row < 0b1000, "{character} row {row:#b} exceeds three bits");
            }
        }
    }

    #[test]
    fn lowercase_resolves_to_the_same_glyph_as_uppercase() {
        assert_eq!(glyph('a'), glyph('A'));
        assert_eq!(glyph('z'), glyph('Z'));
    }

    #[test]
    fn space_is_blank_and_unknown_characters_have_no_glyph() {
        assert_eq!(glyph(' '), Some([0; GLYPH_HEIGHT]));
        assert!(glyph('@').is_none());
    }

    #[test]
    fn distinct_letters_have_distinct_shapes() {
        assert_ne!(glyph('D'), glyph('O'));
        assert_ne!(glyph('E'), glyph('F'));
        assert_ne!(glyph('R'), glyph('P'));
    }

    #[test]
    fn text_width_counts_glyphs_and_the_gaps_between_them() {
        // One glyph is GLYPH_WIDTH wide; each extra glyph adds a column of space.
        assert_eq!(text_width(""), 0);
        assert_eq!(text_width("A"), GLYPH_WIDTH);
        assert_eq!(text_width("AB"), GLYPH_WIDTH * 2 + 1);
        assert_eq!(text_width("ABC"), GLYPH_WIDTH * 3 + 2);
    }
}
