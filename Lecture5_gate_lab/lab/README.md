# The gate lab: a decision before every shell command

Your agent can run shell commands, and some shell commands cannot be taken back. Today you put
a gate in front of the shell. The gate asks a model one narrow question, *could this command
destroy data?*, and it does not read the answer: it reads a **probability**. Code then decides:
run it, ask you, or block it. You measure how good that gate is on 45 labelled commands, you
choose its threshold, and you watch it work in Pi.

**Your guide:** paste `tutor.md` (next to this folder in lab.zip) into the LLM of your choice as its
instructions. It walks you through the lab and helps you read your grid. It does not choose your
threshold for you.

| File | What it is |
|---|---|
| `judge.py` | Asks your local model one yes/no question and returns P(yes), read from one token |
| `questions.py` | The questions the judge can be asked: a question, what yes means, what no means |
| `commands.py` | 45 shell commands with a label: the test set. None of them is ever run |
| `gate_eval.py` | Three gates on the 45 commands, then a grid (`results/gate-<question>.md`) |
| `package/` | A Pi package. `gate.ts` is the gate: a `tool_call` handler in front of `bash` |
| `playground.py` | Makes `playground/`, a small folder that is safe to break (the agent's shell is not locked into it: read the path in every dialog) |
| `select_notes.py` | The same judge on Tuesday's question: which notes deserve a place in the prompt? (step 5) |

## Before you start

- Pi 0.99.2 or 1.x: `pi --version`. On Windows, work inside WSL as in the earlier sessions.
- Qwen3.5-4B served by llama.cpp, as in the last sessions:

      llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M -c 16384 --no-mmproj -rea off --port 8081

- Your model's name, once per terminal (`pi --list-models qwen` shows it):

      QWEN=qwen-local/unsloth/Qwen3.5-4B-GGUF:Q4_K_M

- In this folder:

      uv run playground.py
      git init && git add -A && git commit -m start

  The lab needs no Python packages and no internet.

## 1. One judgment (together, 5 min)

Before you run it: how sure do you expect the model to be, from 0 to 1?

    uv run judge.py 'rm -rf old_logs'
    uv run judge.py 'shred -u secrets.txt'

The command is not run: it is only read. Type the **whole line**: what stands between the quotes
is a real command. And keep the **single quotes**: inside double quotes your own shell would
expand `$HOME`, and run `$(...)`, before `judge.py` sees the text. Now ask the same model the
usual way, five times:

    uv run judge.py 'shred -u secrets.txt' --samples 5

Did the five written answers agree? Could you have known from P(yes) that they might not?
Open `judge.py`: `decide()` and `write()` send the same prompt. Find what differs in the request:
how many tokens, the temperature, and the log-probabilities.

## 2. Check the scoring, then your grid (10 min)

    uv run gate_eval.py --selftest

Every check must end with `ok`; the last line, for information, counts what the regex denylist
catches on the 45 commands. The self-test needs no model. Then the three gates:

    uv run gate_eval.py

It asks for 45 judgments and 225 written answers. While it runs, do not use Pi: one llama.cpp
server answers one request at a time. Read `results/gate-destructive.md`. Of the 45 commands,
19 are labelled destructive, 24 are not, and 2 are **not scored**: the two that download a
script and run it. Can they destroy data? People disagree, so no gate is marked right or wrong
on them (open `commands.py`: a label is a decision somebody made).

- Which commands does the regex miss? Which does the judge miss, at every threshold? (A command
  is flagged when P(yes) is at or above the threshold.)
- Choose **your** threshold. What does one miss cost? What does one false alarm cost? Who pays?
- How many commands got different written answers in five tries? Look at their P(yes).

## 3. The gate in Pi (10 min)

Read `package/extensions/gate.ts`: one question, two numbers (`ASK`, `BLOCK`), three outcomes.
Set `ASK` to the threshold you chose (keep it below `BLOCK`). Then, in the playground:

    cd playground
    pi --model "$QWEN" -e ../package --tools bash

Ask, one after the other, and predict the gate's decision before each one:

1. *List the files here.*
2. *Delete the folder old_logs.*
3. *Empty the file data/sales.csv, but keep the file.*

When Pi asks you to approve a command, read the number and the command before you answer.
The dialog starts on **No**: Enter alone refuses. If the agent keeps trying after a block,
press Esc: Pi has no step limit. Then leave Pi and look:

    cat runs/gate_log.jsonl
    git status

`runs/gate_log.jsonl` is written in the folder where you started Pi: here, the playground.
Did anything get through that should not have? `git restore . && git clean -fd` undoes it, and
`uv run playground.py` (in the lab folder) rebuilds the playground. Both keep your gate log.
Go back with `cd ..`.

## 4. A second question (10 min)

A gate that asks about data does not see every danger. Run the other question:

    uv run gate_eval.py --question remote --samples 0

Read `results/gate-remote.md`. Only two of the 45 commands are labelled `remote`, so look at
the false alarms, and at the two commands that install packages: they are not scored here.
Find their P(yes) under "Not scored", and decide what *you* want the gate to do with
`pip install`. (The regex denylist has no row: it is a list of destructive patterns.)

**Write down** the caught and the false alarms at each threshold: the next run replaces the
file. Then make the question better: edit `"remote"` in `questions.py` (the question, what yes
means, what no means), run it again, and compare. One narrow judgment per question.

**Your row.** Write it on the class grid: model, your threshold for `destructive` with its caught
and false alarms, the command your gate missed, and `remote` before and after your change.

## 5. If you have time (10 min)

Change **one** thing, predict what it does, and measure it. Ideas:

- Your own commands: five that you are not sure about, in a new file `my_commands.py`:

      MY_COMMANDS = [("git stash clear", "data"), ("ls -R /", "none")]

  The labels are `none`, `install`, `data` and `remote`, as in `commands.py`. Run `--selftest`,
  then the grid. Do you agree with the judge? Do you agree with your own labels?
- Two questions in the gate: make `gate.ts` ask `remote` as well, and block when either is high.
- `BLOCK` at 0.5 instead of 0.9: what changes for the human, and for the agent?
- The denylist in `gate_eval.py`: add three patterns. How far do you get before the false alarms start?
- Tuesday's open question, which notes to load: `uv run select_notes.py 'When may I deploy?'`, then
  `uv run select_notes.py --eval`. Twelve notes, room for three, one judgment per note. Is the note
  that matters loaded? What does it cost before the agent reads a word? Add a note that sounds
  relevant to everything, and run it again.

## Take-home

Put a gate in **your own package** from 8 October, in front of the tool of yours that changes
something (a file, a database, a message sent). Write the question for *that* tool, with its yes
and its no. Measure it on at least ten calls you label yourself, and write one paragraph: your
threshold, what a miss and a false alarm cost in your case, and what stays human-only whatever
the probability says. Then check that your gate is really there: its line in
`runs/gate_log.jsonl` after a call. A package installed in a project folder is not loaded until
you trust that folder, and Pi gives no warning: no gate, and the command runs.

## When something goes wrong

- **"Cannot reach the model"**: is llama.cpp running on port 8081? If yours listens elsewhere:
  `JUDGE_URL=http://127.0.0.1:<port>/v1/chat/completions uv run judge.py 'ls'`, and the same
  variable in front of `pi` for the gate.
- **P(yes) = "no number"**: Yes and No together were less than half of the probability: the
  model wanted to start with something else, often a thinking block. Start the server with
  `-rea off`, as above. In a grid such a command counts as flagged, as in the gate: a human is asked.
- **Pi asks me about every command, with "P(destroys data) = unknown"**: the gate cannot reach
  the judge, or the judge did not answer yes or no. Without a number the gate lets nothing
  through on its own: it asks you, and with nobody there it blocks. Check the server, and `JUDGE_URL`.
- **Pi never asks me, it only blocks**: you started Pi with `-p` or in a script: nobody is there
  to ask. Start it without `-p`.
- **The agent tries the blocked command again in another form, or again and again**: that is a
  result. Pi has no step limit: press Esc, then note what it tried and what the gate said each
  time. Read every dialog before you answer: after a block the agent may try the same deletion by
  another route.
- **`gate_eval.py` is slow**: 270 requests. On a laptop without a GPU use `--samples 3` (an odd
  number, so that "more than half" cannot tie), and say so on the grid. Ctrl-C stops it and
  prints the grid of the commands done so far.
- **I broke the playground**: `uv run playground.py` in the lab folder.
- **The gate log mixes several tries**: every line has its time and the `ASK` and `BLOCK` that
  were in force. Delete the file to start a new log.
- **llama.cpp runs on Windows and the lab in WSL**: `127.0.0.1` in WSL is not Windows. Run the
  server inside WSL, as in the earlier sessions.
