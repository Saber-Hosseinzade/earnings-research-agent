"""Citation check, in code: a cited passage ID must be one the retriever returned in this run.

The model writes passage IDs as text, so it can cite an ID it never saw (made
up, misspelled, or remembered from an earlier turn). This check catches that.
It does not judge whether a returned passage supports the claim: that is
scored by the evaluation judge as part of grounding.
"""

from __future__ import annotations

from typing import Any, Iterable


def check_citations(cited: Iterable[str], returned: Iterable[str], covered: bool) -> dict[str, Any]:
    """Compare cited IDs with returned IDs.

    ok is False when any cited ID was not returned, or when the answer says the
    calls cover the question but cites nothing it was given.
    """
    returned_set = set(returned)
    cited_list = list(dict.fromkeys(c.strip() for c in cited if c and c.strip()))  # dedupe, keep order
    verified = [c for c in cited_list if c in returned_set]
    unreturned = [c for c in cited_list if c not in returned_set]
    problems = []
    if unreturned:
        problems.append("cited_not_returned")
    if covered and not verified:
        problems.append("uncited_answer")
    return {"ok": not problems, "problems": problems, "verified": verified, "unreturned": unreturned}
