"""select_notes.py: which notes deserve a place in the prompt? The judge scores each one.

Tuesday's memory loaded the newest notes until the budget was full: simple, and blind to the
request. Here the same judge as in judge.py answers one narrow question for every note,
"would this note help to answer this request?", and code loads the few that score highest.

    uv run select_notes.py 'How much money can we still spend?'      # every note with its score
    uv run select_notes.py 'When may I deploy?' --top 2               # load two notes instead of three
    uv run select_notes.py --eval                                     # eight requests: is the needed note loaded?

One judgment per note and per request: count them before you put this in front of every request.
"""
import argparse
import time

from judge import decide

# A memory file after a few weeks: Tuesday's five rules, and what else got saved.
NOTES = [
    "Client is Halvorsen Marine.",
    "Deploy window opens at 06:40.",
    "Staging server runs on port 8743.",
    "Budget cap is 4,200 euros.",
    "Never write to the client directly: everything goes through Ilse.",
    "Stand-up is at 09:15 in room B.",
    "Ilse is away from 22 to 26 December; Tomas replaces her.",
    "The old staging server was switched off in March.",
    "Invoices go out on the first Monday of the month.",
    "The report template uses the colour #1f4e79 for headings.",
    "Tests must pass before every commit: uv run pytest -q.",
    "The coffee machine on the third floor is the one that works.",
]

RELEVANT = {
    "question": "Would this note help to answer this request?",
    "yes": "the note contains a fact that the answer needs",
    "no": "the request can be answered just as well without this note",
}

# For --eval: a request, and the one note its answer needs (the note's own text, so that you can
# add notes anywhere in the list).
REQUESTS = [
    ("How much money can we still spend on this project?", "Budget cap is 4,200 euros."),
    ("Which port do I connect to for the staging server?", "Staging server runs on port 8743."),
    ("Draft an email that tells the customer about the delay.", "Never write to the client directly: everything goes through Ilse."),
    ("When is the earliest I can deploy tomorrow?", "Deploy window opens at 06:40."),
    ("Who is this project for?", "Client is Halvorsen Marine."),
    ("What must I run before I commit?", "Tests must pass before every commit: uv run pytest -q."),
    ("Who do I ask while Ilse is on holiday?", "Ilse is away from 22 to 26 December; Tomas replaces her."),
    ("When do the invoices go out?", "Invoices go out on the first Monday of the month."),
]


def scores(request: str) -> list[tuple[float | None, str]]:
    """Every note with P(yes), the most useful first. A note without a number goes last."""
    rated = [(decide({"request": request, "note": note}, RELEVANT), note) for note in NOTES]
    if len({p for p, _ in rated}) == 1:
        raise SystemExit(f"The judge gave every note the same answer ({rated[0][0]}): there is nothing to rank.\n"
                         "Is the server the real model, started with -rea off?")
    return sorted(rated, key=lambda pair: 1.0 if pair[0] is None else -pair[0])


def shown(p: float | None) -> str:
    return " none" if p is None else f"{p:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("request", nargs="?", help="what the user asks the agent")
    parser.add_argument("--top", type=int, default=3, help="how many notes are loaded")
    parser.add_argument("--eval", action="store_true", help="the eight test requests: is the needed note loaded?")
    args = parser.parse_args()
    if not args.eval and not args.request:
        parser.error("give a request, or --eval")
    missing = [note for _, note in REQUESTS if note not in NOTES]
    if missing:
        raise SystemExit(f"REQUESTS names a note that is not in NOTES: {missing[0]!r}")

    if not args.eval:
        print(f"{RELEVANT['question']}\n  {args.request}")
        for rank, (p, note) in enumerate(scores(args.request), 1):
            print(f"  {'load' if rank <= args.top else '    '}  {shown(p)}  {note}")
        return

    start = time.time()
    loaded = first = unknown = 0
    for request, needed in REQUESTS:
        ranked = scores(request)
        unknown += sum(p is None for p, _ in ranked)
        rank = [note for _, note in ranked].index(needed) + 1
        loaded += rank <= args.top
        first += rank == 1
        print(f"  rank {rank:>2} of {len(NOTES)}  {'loaded' if rank <= args.top else 'LEFT OUT'}  {request}  ->  {needed}")
    seconds = time.time() - start
    print(f"\nThe needed note is among the {args.top} loaded: {loaded}/{len(REQUESTS)}. It is ranked first: {first}/{len(REQUESTS)}.")
    print(f"{len(REQUESTS) * len(NOTES)} judgments in {seconds:.0f} s: {len(NOTES)} per request, before the agent reads a word."
          + (f" {unknown} of them gave no number." if unknown else ""))
    newest = sum(NOTES.index(needed) >= len(NOTES) - args.top for _, needed in REQUESTS)
    print(f"With room for {args.top} notes, newest first (Tuesday's rule) would have loaded the needed one for {newest}/{len(REQUESTS)} requests.")


if __name__ == "__main__":
    main()
