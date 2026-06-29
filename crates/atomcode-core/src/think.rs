// crates/atomcode-tuix/src/think.rs
pub const THINK_BUF_MAX: usize = 64 * 1024; // 64KB cap to prevent OOM

/// Trailing bytes the stripper retains while waiting for a close tag that may
/// straddle chunk boundaries. Must exceed the longest close tag we recognize —
/// the hashed sentinel `</think_never_used_51bce0c785ca2f68081bfa7d91973934>`
/// is 52 bytes, so 64 leaves headroom.
const CLOSE_TAG_KEEP: usize = 64;

/// Streaming stripper for `<think>...</think>` and `<thinking>...</thinking>`
/// blocks. Case-insensitive, attribute-tolerant, safe across feed boundaries
/// even when a tag is split mid-UTF-8 sequence.
pub struct ThinkStripper {
    /// Holds text that MIGHT be the start of a tag but we can't yet tell.
    /// Only contains bytes at char boundaries.
    carry: String,
    /// When true, we're inside a <think*> block and waiting for </>.
    inside: bool,
}

impl Default for ThinkStripper {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkStripper {
    pub fn new() -> Self {
        Self {
            carry: String::new(),
            inside: false,
        }
    }

    pub fn buffered_bytes(&self) -> usize {
        self.carry.len()
    }

    /// Reset to the pristine state. Call between turns — otherwise an
    /// unclosed `<think>` from a previous turn (model got cancelled, got
    /// an error mid-stream, switched provider, etc.) leaves `inside=true`
    /// and silently swallows every TextDelta of the next turn. Symptom:
    /// user sees blank assistant bubbles even though the provider returned
    /// normal text.
    pub fn reset(&mut self) {
        self.carry.clear();
        self.inside = false;
    }

    /// Feed a chunk, return the visible portion (outside of think blocks).
    pub fn feed(&mut self, delta: &str) -> String {
        // Enforce cap: if carry + delta would exceed THINK_BUF_MAX,
        // we flush the carry as-is (giving up on partial-tag detection)
        // so memory stays bounded.
        if self.carry.len() + delta.len() > THINK_BUF_MAX {
            let mut flushed = std::mem::take(&mut self.carry);
            flushed.push_str(delta);
            if self.inside {
                return String::new(); // still in block, discard
            }
            return flushed;
        }

        self.carry.push_str(delta);
        let mut out = String::new();
        self.drain_into(&mut out);
        out
    }

    fn drain_into(&mut self, out: &mut String) {
        loop {
            if self.inside {
                // Look for closing tag.
                match find_close_tag(&self.carry) {
                    Some((_close_start, close_end)) => {
                        self.carry.drain(..close_end);
                        self.inside = false;
                        // continue loop to scan rest
                    }
                    None => {
                        // keep everything buffered; we might still see </...>
                        // but DROP all but the trailing CLOSE_TAG_KEEP chars to
                        // keep memory bounded under streaming. Must be ≥ the
                        // longest close tag we recognize, else a hashed sentinel
                        // (`</think_never_used_…>`, ~52 chars) split across
                        // chunks would be truncated and never matched.
                        let keep = CLOSE_TAG_KEEP.min(self.carry.len());
                        let drop_end = self.carry.len() - keep;
                        let safe = prev_boundary(&self.carry, drop_end);
                        self.carry.drain(..safe);
                        return;
                    }
                }
            } else {
                match scan_outside(&self.carry) {
                    OutsideScan::None => {
                        out.push_str(&self.carry);
                        self.carry.clear();
                        return;
                    }
                    OutsideScan::Open { start, end } => {
                        out.push_str(&self.carry[..start]);
                        self.carry.drain(..end);
                        self.inside = true;
                    }
                    OutsideScan::StrayClose { start, end } => {
                        // A close tag with no matching open: the open tag + body
                        // went out a SEPARATE reasoning channel and only the
                        // closing sentinel straggled into visible text (the
                        // `</think_never_used_…>` leak). Drop it as noise and
                        // keep scanning the remainder.
                        out.push_str(&self.carry[..start]);
                        self.carry.drain(..end);
                    }
                    OutsideScan::PartialAt(pos) => {
                        out.push_str(&self.carry[..pos]);
                        self.carry.drain(..pos);
                        return;
                    }
                }
            }
        }
    }
}

/// Result of scanning outside-of-block text for the next think tag.
enum OutsideScan {
    None,
    /// A complete open tag (`<think…>`): enter the block.
    Open { start: usize, end: usize },
    /// A complete close tag with no matching open: a stray reasoning-channel
    /// delimiter that leaked into visible text. Drop it.
    StrayClose { start: usize, end: usize },
    /// A prefix that could still grow into an open OR close tag — hold back.
    PartialAt(usize),
}

/// Scan outside-block text for the next think tag (open or stray close), or a
/// partial prefix of one straddling the feed boundary. Case-insensitive.
fn scan_outside(s: &str) -> OutsideScan {
    let mut search_start = 0;
    while let Some(lt) = s[search_start..].find('<') {
        let abs = search_start + lt;
        let rest = &s[abs..];
        if let Some(end) = parse_open_tag(rest) {
            return OutsideScan::Open { start: abs, end: abs + end };
        }
        if let Some(end) = parse_close_tag(rest) {
            return OutsideScan::StrayClose { start: abs, end: abs + end };
        }
        if could_be_partial_tag(rest) {
            return OutsideScan::PartialAt(abs);
        }
        search_start = abs + 1;
    }
    OutsideScan::None
}

/// Whether `rest` (starting at a `<`) is a prefix that could still grow into a
/// `<think…>` open tag or a `</think…>` close tag once more bytes arrive — i.e.
/// a tag split across the feed boundary, with no closing `>` yet.
fn could_be_partial_tag(rest: &str) -> bool {
    let lower: String = rest.chars().map(|c| c.to_ascii_lowercase()).collect();
    if lower.contains('>') {
        return false; // a `>` is present, so this `<…>` is already complete/decided
    }
    // Still typing out the opening of a (possibly hashed) think/close tag.
    "<thinking".starts_with(lower.as_str())
        || "</thinking".starts_with(lower.as_str())
        || lower.starts_with("<think")
        || lower.starts_with("</think")
}

/// If `s` starts with a complete `<think...>` or `<thinking...>` open tag,
/// returns byte end position (just after `>`).
///
/// NOTE: attribute values are matched by finding the next literal `>`.
/// A quoted attribute containing a literal `>` (e.g. `<think foo="a>b">`)
/// would confuse this parser. LLM-generated think tags in practice carry
/// no meaningful attributes, so this trade-off is intentional.
fn parse_open_tag(s: &str) -> Option<usize> {
    if !s.starts_with('<') {
        return None;
    }
    let lower_head: String = s.chars().take(10).map(|c| c.to_ascii_lowercase()).collect();
    let name_end = if lower_head.starts_with("<thinking") {
        9
    } else if lower_head.starts_with("<think") {
        6
    } else {
        return None;
    };
    let after = &s[name_end..];
    let first = after.chars().next()?;
    // Plain `<think>` / `<thinking>`.
    if first == '>' {
        return Some(name_end + 1);
    }
    // Attribute form `<think foo="bar">`, OR a hashed-sentinel name some chat
    // templates use to delimit the reasoning channel — e.g.
    // `<think_never_used_51bce0c785ca2f68081bfa7d91973934>`. In the sentinel
    // case the bytes after `<think` are name-continuation chars (`_`, `-`,
    // alphanumerics) up to the closing `>`. Keying on `_`/`-` (not any letter)
    // keeps plain words like `<thinkable>` / `<thinktank>` from being mistaken
    // for think tags.
    if first.is_ascii_whitespace() || first == '_' || first == '-' {
        if let Some(gt) = after.find('>') {
            return Some(name_end + gt + 1);
        }
    }
    None
}

/// If `s` STARTS with a complete close tag — `</think>`, `</thinking>`, or a
/// hashed sentinel `</think_never_used_…>` — returns its byte length. Keying
/// the post-`</think` byte on `>` / `ing>` / `_…` / `-…` (not any letter) keeps
/// `</thinktank>` and similar from being mistaken for a think close tag.
fn parse_close_tag(s: &str) -> Option<usize> {
    let lower_head: String = s.chars().take(11).map(|c| c.to_ascii_lowercase()).collect();
    if lower_head.starts_with("</thinking") {
        // Close tags carry no attributes — only a bare `</thinking>`.
        let after = &s["</thinking".len()..];
        return after.starts_with('>').then_some("</thinking>".len());
    }
    if lower_head.starts_with("</think") {
        let after = &s["</think".len()..];
        let first = after.chars().next()?;
        if first == '>' {
            return Some("</think>".len());
        }
        if first == '_' || first == '-' {
            if let Some(gt) = after.find('>') {
                return Some("</think".len() + gt + 1);
            }
        }
    }
    None
}

/// Find the next `</think>`, `</thinking>`, or hashed-sentinel close tag
/// (e.g. `</think_never_used_51bce0…>`), case-insensitive. Returns the byte
/// range of the EARLIEST valid close tag.
fn find_close_tag(s: &str) -> Option<(usize, usize)> {
    let lower: String = s.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut from = 0;
    while let Some(rel) = lower[from..].find("</think") {
        let start = from + rel;
        if let Some(len) = parse_close_tag(&s[start..]) {
            return Some((start, start + len));
        }
        from = start + "</think".len();
    }
    None
}

fn prev_boundary(s: &str, mut idx: usize) -> usize {
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tags_passes_through() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("hello world"), "hello world");
    }

    #[test]
    fn complete_block_in_one_feed() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("a<think>secret</think>b"), "ab");
    }

    #[test]
    fn tag_split_across_feeds() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("hello <thi"), "hello ");
        assert_eq!(s.feed("nk>secret</think> world"), " world");
    }

    #[test]
    fn utf8_boundary_at_feed_edge_no_panic() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("abc<thi"), "abc");
        assert_eq!(s.feed("nk>密</think>你好"), "你好");
    }

    #[test]
    fn case_insensitive_tag() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("<THINK>a</THINK>b"), "b");
        let mut s2 = ThinkStripper::new();
        assert_eq!(s2.feed("<Think>a</Think>b"), "b");
    }

    #[test]
    fn thinking_tag_also_stripped() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("<thinking>a</thinking>b"), "b");
    }

    #[test]
    fn tag_with_attributes() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("<think key=\"v\">a</think>b"), "b");
    }

    #[test]
    fn unclosed_block_capped_at_buf_limit() {
        let mut s = ThinkStripper::new();
        let junk = "x".repeat(100_000);
        let input = format!("<think>{}", junk);
        let _ = s.feed(&input);
        assert!(s.buffered_bytes() <= THINK_BUF_MAX);
    }

    #[test]
    fn literal_angle_bracket_outside_tag_preserved() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("a < b > c"), "a < b > c");
    }

    #[test]
    fn multiple_blocks() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("a<think>x</think>b<think>y</think>c"), "abc");
    }

    /// Regression: an unclosed `<think>` from a previous turn would leave
    /// `inside=true` and swallow the entire next turn's text. The real-world
    /// trigger is a provider switch mid-turn (e.g. GLM → Kimi): GLM embeds
    /// thinking as `<think>…</think>` in content, Kimi routes it through
    /// `reasoning_content` (plain content with no `<think>` tag). If the
    /// GLM turn cancels with an open tag and no one calls `reset()`, every
    /// Kimi TextDelta afterward disappears — user sees blank assistant
    /// bubbles while datalog shows the LLM actually returned text.
    #[test]
    fn reset_clears_stuck_inside_state() {
        let mut s = ThinkStripper::new();
        // Turn 1: opens a think block but never closes it (stream ended /
        // got cancelled).
        let _ = s.feed("prefix <think>still thinking when we got cut");
        // Turn 2 from a different provider that doesn't use <think> tags.
        // Without reset, this text gets swallowed.
        assert_eq!(
            s.feed("hello from the next model"),
            "",
            "without reset, text leaks through the stuck inside=true state",
        );
        // Apply the fix.
        s.reset();
        // Turn 3: now the stripper is pristine and plain text passes.
        assert_eq!(
            s.feed("hello from the next model"),
            "hello from the next model"
        );
    }

    #[test]
    fn reset_from_pristine_state_is_a_noop() {
        // Calling reset on a fresh stripper shouldn't break subsequent feeds.
        let mut s = ThinkStripper::new();
        s.reset();
        assert_eq!(s.feed("plain text"), "plain text");
    }

    #[test]
    fn reset_clears_partial_carry_at_feed_boundary() {
        // A feed that ends mid-tag leaves bytes in `carry` awaiting the
        // rest of the tag. Reset should flush that too so the next turn
        // starts clean.
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("hello <thi"), "hello "); // carry now holds "<thi"
        assert!(s.buffered_bytes() > 0);
        s.reset();
        assert_eq!(s.buffered_bytes(), 0);
        // Without reset the next feed would try to complete the tag;
        // with reset, "<think>" is treated as the start of a new block.
        assert_eq!(s.feed("not a tag: <3"), "not a tag: <3");
    }

    const SENTINEL: &str = "think_never_used_51bce0c785ca2f68081bfa7d91973934";

    /// Some chat templates wrap the reasoning channel in a hashed sentinel tag
    /// instead of `<think>`. The whole block must be stripped like any other.
    #[test]
    fn hashed_sentinel_block_stripped() {
        let mut s = ThinkStripper::new();
        let input = format!("a<{SENTINEL}>secret reasoning</{SENTINEL}>b");
        assert_eq!(s.feed(&input), "ab");
    }

    /// Regression for the reported leak: the reasoning content was routed away
    /// but the closing sentinel tag straggled into the visible channel and
    /// rendered verbatim in the webui. A stray close tag must be swallowed, not
    /// shown.
    #[test]
    fn hashed_sentinel_stray_close_tag_swallowed() {
        let mut s = ThinkStripper::new();
        let input = format!("…更符合手机APP场景</{SENTINEL}>");
        assert_eq!(s.feed(&input), "…更符合手机APP场景");
    }

    /// The hashed close tag (~52 bytes) is far longer than `</thinking>`. When
    /// it arrives split across feeds the stripper must still match it — this
    /// guards the `CLOSE_TAG_KEEP` trailing-buffer size.
    #[test]
    fn hashed_sentinel_close_tag_split_across_feeds() {
        let mut s = ThinkStripper::new();
        let open = format!("<{SENTINEL}>");
        assert_eq!(s.feed(&format!("x{open}secret ")), "x");
        // Feed the long close tag one byte at a time; nothing should leak and
        // the block must close exactly when the final `>` arrives.
        let close = format!("</{SENTINEL}>");
        let bytes: Vec<char> = close.chars().collect();
        for (i, c) in bytes.iter().enumerate() {
            let out = s.feed(&c.to_string());
            assert_eq!(out, "", "close-tag byte {i} ({c:?}) must not leak");
        }
        assert_eq!(s.feed("after"), "after");
    }

    /// Plain words that merely start with "think" must NOT be treated as think
    /// tags — only `>`, whitespace (attributes), `_`, or `-` may follow `<think`.
    #[test]
    fn non_think_words_not_stripped() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("<thinktank>ideas</thinktank>"), "<thinktank>ideas</thinktank>");
        let mut s2 = ThinkStripper::new();
        assert_eq!(s2.feed("a <thinkable> b"), "a <thinkable> b");
    }
}
