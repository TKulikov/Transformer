"""Multi-dialogue corpus generator (plan: docs/PLAN-fullscale-tests.md, step 1).

A dialogue is *data*, not code: :class:`CorpusSpec` (topic × persona ×
scenario × length × seed) deterministically expands into a
:class:`~tmem.qa.DialogSession`. Axes of variation:

* **topic** — six domains, each with its own fact pool, twin pairs,
  holdout facts (never planted, for unanswerable probes), distractors
  and small talk;
* **persona** — four user styles (terse, chatty, sloppy, formal); styles
  re-wrap statements but never touch fact values, so keyword scoring
  stays exact;
* **scenario** — baseline / updates / adversarial (twin facts) /
  distractors / multifact (2 facts per turn);
* **length / lag** — target turn count and probes spread over lag zones
  short (1–5), mid (6–20), long (21–50), vlong (51+), not only at the end;
* **probe kind** — direct, paraphrase, multihop, temporal (old value
  expected after an update) and unanswerable (hallucination gauge).

All keyword pools are validated for substring collisions
(:func:`validate_topics`), and every generated session is validated
(:func:`validate_session`) — an expected answer must be unique inside
its dialogue and planted before the probe.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .qa import DialogSession, Probe

# ------------------------------------------------------------------ data model


@dataclass(frozen=True)
class FactSpec:
    """One plantable fact with its probe texts and optional supersession."""

    key: str
    statement: str                       # contains every expected keyword verbatim
    expected: tuple[str, ...]
    question: str
    paraphrase: str
    update: str | None = None            # statement that supersedes the fact
    new_expected: tuple[str, ...] = ()
    temporal_question: str | None = None  # asks the OLD value after the update


@dataclass(frozen=True)
class Topic:
    """A thematic domain: facts, twins, holdout, distractors, small talk."""

    name: str
    label: str                            # human-readable, used in openers
    facts: tuple[FactSpec, ...]
    twins: tuple[tuple[FactSpec, FactSpec], ...]
    holdout: tuple[FactSpec, ...]         # never planted → unanswerable probes
    distractors: tuple[str, ...]          # semantically close, value-free
    fillers: tuple[str, ...]


@dataclass(frozen=True)
class Persona:
    """A user communication style. Templates wrap text, never alter values."""

    name: str
    opener: str                           # contains {topic}
    plant_templates: tuple[str, ...]      # contain {s}
    question_templates: tuple[str, ...]   # contain {q}
    fillers: tuple[str, ...]              # complete short lines

    def style_plant(self, text: str, rng: random.Random) -> str:
        tpl = rng.choice(self.plant_templates)
        if "{s_lower}" in tpl:
            return tpl.replace("{s_lower}", text.lower())
        return tpl.format(s=text)

    def style_question(self, text: str, rng: random.Random) -> str:
        tpl = rng.choice(self.question_templates)
        if "{q_lower}" in tpl:
            return tpl.replace("{q_lower}", text.lower())
        if not tpl.startswith("{q}"):
            text = text[0].lower() + text[1:]  # question continues the persona's prefix
        return tpl.format(q=text)


SCENARIOS = ("baseline", "updates", "adversarial", "distractors", "multifact")

# lag zone name → (min_lag, max_lag); vlong only fits long dialogues
LAG_ZONES = (("short", 1, 5), ("mid", 6, 20), ("long", 21, 50), ("vlong", 51, 10**9))

GENERIC_FILLERS = (
    "Tell me more about that.",
    "What else should I keep in mind?",
    "Keep going, please.",
    "Interesting — and what happened next?",
)


@dataclass(frozen=True)
class CorpusSpec:
    """One dialogue of the corpus, fully reproducible from these fields."""

    spec_id: str
    topic: str
    persona: str
    scenario: str
    target_turns: int = 100
    seed: int = 0
    n_facts: int = 10
    probes_per_zone: int = 4
    n_unanswerable: int = 2
    n_temporal: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CorpusSpec":
        known = {f for f in CorpusSpec.__dataclass_fields__}
        return CorpusSpec(**{k: v for k, v in d.items() if k in known})


# ------------------------------------------------------------------ topics

TOPICS: tuple[Topic, ...] = (
    Topic(
        name="work",
        label="my work and IT project",
        facts=(
            FactSpec("team-lead", "My team lead is called Margarita.", ("margarita",),
                     "Who leads my team?", "What is the name of my team leader?"),
            FactSpec("sprint", "Our current sprint ends on Thursday.", ("thursday",),
                     "When does my sprint end?", "On which day does the current sprint finish?"),
            FactSpec("deploy", "We deploy the service via a canary release.", ("canary",),
                     "How do we deploy the service?", "Which release method does my team use?"),
            FactSpec("stack", "The backend is written in Golang.", ("golang",),
                     "Which language is the backend written in?", "What is the backend stack?"),
            FactSpec("ticket", "The epic ticket number is PRJ-482.", ("482",),
                     "What is the epic ticket number?", "Which ticket tracks the epic?"),
            FactSpec("retro", "Our retrospective starts at 15:30.", ("15:30",),
                     "When does the retrospective start?", "At what time is the retro?"),
            FactSpec("intern", "Our new intern is called Pavel.", ("pavel",),
                     "What is the name of our intern?", "Who is the new intern?"),
            FactSpec("office", "The office is on the twelfth floor.", ("twelfth",),
                     "Which floor is the office on?", "Where is the office located?"),
            FactSpec("deadline", "The release deadline is June 3.", ("june",),
                     "When is the release deadline?", "By which date must we release?",
                     update="The release slipped to July 8.", new_expected=("july",),
                     temporal_question="What was the release deadline before it slipped?"),
            FactSpec("reviewer", "Dana reviews my pull requests.", ("dana",),
                     "Who reviews my pull requests?", "Whose review do I wait for on PRs?",
                     update="Now Igor reviews my pull requests.", new_expected=("igor",),
                     temporal_question="Who reviewed my pull requests before the change?"),
        ),
        twins=(
            (FactSpec("staging", "Our staging server is called Falcon.", ("falcon",),
                      "What is the name of the staging server?", "Which server hosts staging?"),
             FactSpec("prod", "Our production server is called Phoenix.", ("phoenix",),
                      "What is the name of the production server?", "Which server hosts production?")),
        ),
        holdout=(
            FactSpec("ci", "Our continuous integration runs on Buildkite.", ("buildkite",),
                     "Which CI platform do we use?", "Where do the pipelines run?"),
            FactSpec("hq", "The company headquarters is in Limassol.", ("limassol",),
                     "Where is the company HQ?", "Which city hosts the head office?"),
        ),
        distractors=(
            "Colleagues keep discussing deployment pipelines in every chat.",
            "We had a long call about quarterly planning and roadmaps.",
            "The office coffee machine broke again this week.",
        ),
        fillers=(
            "How should I phrase this in a standup update?",
            "What else is worth tracking for the team?",
            "Tell me more about engineering culture in such teams.",
        ),
    ),
    Topic(
        name="travel",
        label="my upcoming trip",
        facts=(
            FactSpec("dest", "We fly to Lisbon in the spring.", ("lisbon",),
                     "Where are we flying?", "Which city is the destination?"),
            FactSpec("flight", "My flight number is TP-1264.", ("1264",),
                     "What is my flight number?", "Which flight am I booked on?"),
            FactSpec("hotel", "The hotel is called Miradouro.", ("miradouro",),
                     "What is the name of the hotel?", "Where are we staying?",
                     update="Actually we rebooked to Alfama Suites.", new_expected=("alfama",),
                     temporal_question="Which hotel did we book before the rebooking?"),
            FactSpec("fdate", "The flight departs on May 12.", ("may 12",),
                     "When does the flight depart?", "What is the departure date?",
                     update="The flight moved to May 19.", new_expected=("may 19",),
                     temporal_question="What was the departure date before the move?"),
            FactSpec("insurance", "My travel insurance policy is TRV-993.", ("993",),
                     "What is the insurance policy number?", "Which policy covers the trip?"),
            FactSpec("guide", "The local guide is named Tomas.", ("tomas",),
                     "What is the name of the guide?", "Who is our local guide?"),
            FactSpec("budget", "The trip budget is 1400 euro.", ("1400",),
                     "What is the trip budget?", "How much money did we plan to spend?"),
            FactSpec("train", "We take the night train to Porto.", ("porto",),
                     "Where does the night train go?", "Which city does the train reach?"),
            FactSpec("camera", "I am taking the old Zenit camera along.", ("zenit",),
                     "Which camera am I taking?", "What photo gear comes with us?"),
            FactSpec("visa", "The visa appointment is at the consulate.", ("consulate",),
                     "Where is the visa appointment?", "Which office handles the visa?"),
        ),
        twins=(
            (FactSpec("hostel", "The cheap hostel stands in Braga.", ("braga",),
                      "In which city is the hostel?", "Where is the hostel located?"),
             FactSpec("apartments", "The rented apartments are in Coimbra.", ("coimbra",),
                      "In which city are the apartments?", "Where did we rent apartments?")),
        ),
        holdout=(
            FactSpec("return", "The return flight is FR-8871.", ("8871",),
                     "What is the return flight number?", "Which flight brings us back?"),
            FactSpec("show", "We booked a fado dinner show.", ("fado",),
                     "What evening show did we book?", "Which performance is scheduled?"),
        ),
        distractors=(
            "I keep re-checking the packing list every evening.",
            "The weather forecast for the coast changes twice a day.",
            "Friends recommend trying the local pastries and markets.",
        ),
        fillers=(
            "What else should go into the itinerary?",
            "Tell me something useful about long trips.",
            "How do people usually plan city walks?",
        ),
    ),
    Topic(
        name="health",
        label="my health routine",
        facts=(
            FactSpec("doctor", "My cardiologist is Doctor Ivanova.", ("ivanova",),
                     "Who is my cardiologist?", "Which doctor checks my heart?"),
            FactSpec("meds", "I take 25 mg of lisinopril daily.", ("lisinopril",),
                     "Which medication do I take?", "What pills are in my daily routine?",
                     update="The doctor switched me to enalapril.", new_expected=("enalapril",),
                     temporal_question="Which medication did I take before the switch?"),
            FactSpec("checkup", "The checkup is on Friday at nine.", ("friday",),
                     "When is the checkup?", "On which day do I visit the clinic?",
                     update="The checkup moved to Monday.", new_expected=("monday",),
                     temporal_question="When was the checkup scheduled before the move?"),
            FactSpec("diet", "I stopped eating anything with lactose.", ("lactose",),
                     "What do I avoid in my diet?", "Which ingredient did I cut out?"),
            FactSpec("steps", "My daily goal is 8000 steps.", ("8000",),
                     "What is my daily step goal?", "How many steps do I aim for?"),
            FactSpec("sleep", "I go to bed at 23:00 on weekdays.", ("23:00",),
                     "When do I go to bed?", "What is my weekday bedtime?"),
            FactSpec("water", "I drink two liters of water a day.", ("two liters",),
                     "How much water do I drink daily?", "What is my water intake?"),
            FactSpec("vitamin", "I take vitamin D3 during winter.", ("d3",),
                     "Which vitamin do I take in winter?", "What supplement helps me in winter?"),
            FactSpec("trainer", "My physio trainer is called Oleg.", ("oleg",),
                     "Who is my physio trainer?", "Which trainer runs my exercises?"),
            FactSpec("clinic", "The clinic is on Sirenevy Boulevard.", ("sirenevy",),
                     "Where is the clinic?", "Which street hosts the clinic?"),
        ),
        twins=(
            (FactSpec("therapist", "My therapist is Anna.", ("anna",),
                      "Who is my therapist?", "Which specialist do I talk to weekly?"),
             FactSpec("dentist", "My dentist is Greta.", ("greta",),
                      "Who is my dentist?", "Which doctor treats my teeth?")),
        ),
        holdout=(
            FactSpec("blood", "My blood type is AB negative.", ("ab negative",),
                     "What is my blood type?", "Which blood group do I have?"),
            FactSpec("scan", "I had an MRI scan scheduled.", ("mri",),
                     "Which scan was scheduled?", "What imaging did the doctor order?"),
        ),
        distractors=(
            "Sleep quality changed a lot since the weather turned cold.",
            "Colleagues at the gym keep swapping workout programs.",
            "I read an article about morning routines, mostly common sense.",
        ),
        fillers=(
            "What else helps to stay in shape?",
            "Tell me more about healthy habits.",
            "How do people keep such routines consistent?",
        ),
    ),
    Topic(
        name="finance",
        label="my shopping and finances",
        facts=(
            FactSpec("bank", "My bank is called Meridian.", ("meridian",),
                     "What is the name of my bank?", "Where do I keep my accounts?"),
            FactSpec("card", "My card ends with 4291.", ("4291",),
                     "What are the last digits of my card?", "Which card do I use?"),
            FactSpec("bill", "The electricity bill this month is 63 euro.", ("63 euro",),
                     "How much is the electricity bill?", "What did I pay for power?"),
            FactSpec("savings", "I transfer 200 euro into savings monthly.", ("200",),
                     "How much do I save monthly?", "What is my monthly savings transfer?"),
            FactSpec("subscription", "My music subscription costs 9.99 a month.", ("9.99",),
                     "How much is the music subscription?", "What do I pay for music monthly?",
                     update="I downgraded to the yearly plan for 79 euro.", new_expected=("79",),
                     temporal_question="What was the monthly music subscription price before the downgrade?"),
            FactSpec("tariff", "My mobile tariff is called Storm.", ("storm",),
                     "What is my mobile tariff?", "Which plan does my phone use?",
                     update="I switched to the Calm tariff.", new_expected=("calm",),
                     temporal_question="Which tariff did I use before the switch?"),
            FactSpec("laptop", "I bought the laptop for 1180 euro.", ("1180",),
                     "How much did the laptop cost?", "What was the price of the laptop?"),
            FactSpec("broker", "My broker account is at Vector.", ("vector",),
                     "Where is my broker account?", "Which broker do I use?"),
            FactSpec("rent", "The rent is 540 euro per month.", ("540",),
                     "How much is the monthly rent?", "What do I pay for the apartment?"),
            FactSpec("cashback", "The card gives five percent cashback.", ("cashback",),
                     "What bonus does the card give?", "Which perk comes with the card?"),
        ),
        twins=(
            (FactSpec("visa", "The Visa card expires in April.", ("april",),
                      "When does the Visa card expire?", "Which month ends the Visa validity?"),
             FactSpec("mc", "The Mastercard expires in August.", ("august",),
                      "When does the Mastercard expire?", "Which month ends the Mastercard validity?")),
        ),
        holdout=(
            FactSpec("fund", "I keep an emergency fund of 3000 euro.", ("3000",),
                     "How large is the emergency fund?", "What is my financial cushion?"),
            FactSpec("safe", "The safe box code is 7734.", ("7734",),
                     "What is the safe box code?", "Which combination opens the safe?"),
        ),
        distractors=(
            "Prices in the supermarket next door keep creeping up.",
            "I compared several loyalty programs without deciding.",
            "The news keeps predicting changes in interest rates.",
        ),
        fillers=(
            "What else should I count in the budget?",
            "Tell me more about tracking expenses.",
            "How do people plan big purchases?",
        ),
    ),
    Topic(
        name="study",
        label="my studies",
        facts=(
            FactSpec("course", "I enrolled in the linear algebra course.", ("algebra",),
                     "Which course did I enroll in?", "What subject am I taking this term?"),
            FactSpec("thesis", "My thesis topic is spectral clustering.", ("spectral",),
                     "What is my thesis topic?", "Which subject is the thesis about?"),
            FactSpec("advisor", "My advisor is Professor Ertel.", ("ertel",),
                     "Who is my advisor?", "Which professor supervises me?",
                     update="Now Professor Vogel supervises me.", new_expected=("vogel",),
                     temporal_question="Who was my advisor before the change?"),
            FactSpec("library", "I study on the third floor of the library.", ("third floor",),
                     "Where do I usually study?", "Which library floor is mine?"),
            FactSpec("exam", "The statistics exam is on December 20.", ("december",),
                     "When is the statistics exam?", "Which date is the exam on?",
                     update="The exam was rescheduled to January 9.", new_expected=("january",),
                     temporal_question="When was the exam scheduled before the rescheduling?"),
            FactSpec("group", "My study group meets on Wednesdays.", ("wednesdays",),
                     "When does the study group meet?", "On which day do we gather to study?"),
            FactSpec("german", "I practice German twice a week.", ("german",),
                     "Which language do I practice?", "What am I learning besides the major?"),
            FactSpec("grant", "The small grant covers the conference fee.", ("grant",),
                     "What covers the conference fee?", "Which funding pays for the trip?"),
            FactSpec("lab", "The lab work starts at ten in the morning.", ("ten",),
                     "When does the lab work start?", "At what hour is the lab session?"),
            FactSpec("seminar-notes", "I share my notes through the group chat.", ("notes",),
                     "How do I share my study materials?", "What do I post to the group?"),
        ),
        twins=(
            (FactSpec("morning-sem", "The morning seminar is in room 402.", ("402",),
                      "Which room hosts the morning seminar?", "Where is the seminar before lunch?"),
             FactSpec("evening-sem", "The evening seminar is in room 407.", ("407",),
                      "Which room hosts the evening seminar?", "Where is the seminar after dinner?")),
        ),
        holdout=(
            FactSpec("chess", "I signed up for the university chess club.", ("chess",),
                     "Which club did I join?", "What board game do I play at university?"),
            FactSpec("summer", "The summer school takes place in Trieste.", ("trieste",),
                     "Where is the summer school?", "Which city hosts the summer program?"),
        ),
        distractors=(
            "The lecture hall projector flickers every second slide.",
            "Classmates argue about which elective to pick next term.",
            "The campus wi-fi drops during the evening rush.",
        ),
        fillers=(
            "What else should a student keep track of?",
            "Tell me more about preparing for exams.",
            "How do people organize thesis work?",
        ),
    ),
    Topic(
        name="home",
        label="my home and hobbies",
        facts=(
            FactSpec("repair", "The plumber finally fixed the kitchen sink.", ("plumber",),
                     "Who fixed the kitchen sink?", "Which worker repaired the sink?"),
            FactSpec("sofa", "We ordered a green velvet sofa.", ("velvet",),
                     "What kind of sofa did we order?", "Which material is the new sofa?",
                     update="The sofa was replaced by a beige one.", new_expected=("beige",),
                     temporal_question="Which sofa did we order before the replacement?"),
            FactSpec("garden", "I planted jasmine near the fence.", ("jasmine",),
                     "What did I plant near the fence?", "Which flower grows by the fence?",
                     update="I replanted lavender along the fence instead.", new_expected=("lavender",),
                     temporal_question="What grew near the fence before the replanting?"),
            FactSpec("bike", "My bike is a gravel model called Falcon R2.", ("falcon r2",),
                     "Which bike do I ride?", "What is my bike model?"),
            FactSpec("piano", "I practice piano thirty minutes every day.", ("piano",),
                     "Which instrument do I practice?", "What do I play daily?"),
            FactSpec("lamp", "The new lamp is from Nordlys.", ("nordlys",),
                     "Which brand is the new lamp?", "Where did the lamp come from?"),
            FactSpec("paint", "We painted the bedroom terracotta.", ("terracotta",),
                     "What color is the bedroom now?", "Which paint did we choose?"),
            FactSpec("boardgame", "We play Terraforming Mars on Sundays.", ("terraforming",),
                     "Which board game do we play?", "What game is our Sunday ritual?"),
            FactSpec("shelf", "The hallway shelf is made of walnut.", ("walnut",),
                     "What is the hallway shelf made of?", "Which wood is the shelf?"),
            FactSpec("curtains", "The curtains were sewn by a local atelier.", ("atelier",),
                     "Who made the curtains?", "Where were the curtains sewn?"),
        ),
        twins=(
            (FactSpec("attic-shelf", "The attic shelf is oak.", ("oak",),
                      "Which wood is the attic shelf?", "What is the attic shelf made of?"),
             FactSpec("balcony-shelf", "The balcony shelf is teak.", ("teak",),
                      "Which wood is the balcony shelf?", "What is the balcony shelf made of?")),
        ),
        holdout=(
            FactSpec("parrot", "We adopted a parrot named Kiko.", ("kiko",),
                     "What is the name of the parrot?", "Which bird did we adopt?"),
            FactSpec("vacuum", "We ordered a robot vacuum XR-70.", ("xr-70",),
                     "Which robot vacuum did we order?", "What model is the vacuum?"),
        ),
        distractors=(
            "Neighbors renovate their flat and drill every weekend.",
            "The yard cats sleep on the warm car hood again.",
            "I keep reorganizing the tool boxes in the garage.",
        ),
        fillers=(
            "What else makes a house cozy?",
            "Tell me more about weekend projects.",
            "How do people combine hobbies and chores?",
        ),
    ),
)

PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="terse",
        opener="Quick notes about {topic}. Write them down.",
        plant_templates=("{s}", "Note: {s}", "{s} Remember that."),
        question_templates=("{q}", "Question: {q}"),
        fillers=("Next.", "Continue.", "Go on.", "Anything else?"),
    ),
    Persona(
        name="chatty",
        opener="Oh, I have SO much to tell you about {topic}!",
        plant_templates=(
            "Oh, by the way — {s} Can you imagine?",
            "Guess what! {s} I was going to tell you all morning.",
            "So anyway, {s} Isn't that exciting?",
        ),
        question_templates=("Quick question! {q}", "By the way, {q}", "Tell me, {q}"),
        fillers=(
            "That reminds me of a hundred stories, but you tell me something else!",
            "I could talk about this for hours! What else is on your mind?",
            "Anyway, life is full of surprises, right? Keep going!",
        ),
    ),
    Persona(
        name="sloppy",
        opener="yo lets talk about {topic} or smth",
        plant_templates=("btw {s_lower}", "uhh {s_lower}", "{s_lower} i guess"),
        question_templates=("{q_lower}??", "so {q_lower}", "{q_lower} btw"),
        fillers=("anyway... more stuff", "idk what else to say, keep going", "uhh next thing"),
    ),
    Persona(
        name="formal",
        opener="This conversation concerns {topic}. Please record the details.",
        plant_templates=(
            "Please take note of the following: {s}",
            "Kindly register this information: {s}",
            "{s} Please add this to the record.",
        ),
        question_templates=("Kindly answer: {q}", "Please provide the following: {q}", "I would like to clarify: {q}"),
        fillers=(
            "Kindly continue the discussion.",
            "Please elaborate on the previous point.",
            "I would appreciate further details.",
        ),
    ),
)

TOPIC_NAMES = tuple(t.name for t in TOPICS)
PERSONA_NAMES = tuple(p.name for p in PERSONAS)


# ------------------------------------------------------------------ generator


def _fact_keywords(f: FactSpec) -> list[str]:
    return [k.lower() for k in (*f.expected, *f.new_expected)]


def _free_near(slots: list, occupied: set[int], want: int, lo: int, hi: int) -> int | None:
    """Find the nearest free turn to ``want`` inside ``[lo, hi]``."""
    for delta in range(0, hi - lo + 2):
        for t in ((want + delta), (want - delta)):
            if lo <= t <= hi and t < len(slots) and t not in occupied and slots[t] is None:
                return t
    return None


def build_corpus_session(
    spec: CorpusSpec,
    topics: tuple[Topic, ...] = TOPICS,
    personas: tuple[Persona, ...] = PERSONAS,
) -> DialogSession:
    """Expand ``spec`` into a deterministic :class:`DialogSession`.

    The same spec always produces byte-identical turns and probes; the seed
    is derived from ``spec.seed`` and ``spec.spec_id``.
    """
    topic = next(t for t in topics if t.name == spec.topic)
    persona = next(p for p in personas if p.name == spec.persona)
    rng = random.Random(f"corpus:{spec.seed}:{spec.spec_id}")
    T = spec.target_turns
    slots: list[str | None] = [None] * T
    occupied: set[int] = {0}
    slots[0] = persona.opener.format(topic=topic.label)

    # --- fact selection -----------------------------------------------------
    pool = list(topic.facts)
    chosen: list[FactSpec] = []
    if spec.scenario == "adversarial" and topic.twins:
        chosen.extend(topic.twins[0])
    if spec.scenario == "updates":
        upd = [f for f in pool if f.update]
        chosen.extend(rng.sample(upd, min(spec.n_temporal, len(upd))))
    rest = [f for f in pool if f not in chosen]
    n_more = max(0, min(spec.n_facts, len(pool) + len(chosen)) - len(chosen))
    chosen.extend(rng.sample(rest, min(n_more, len(rest))))

    # --- planting -----------------------------------------------------------
    plants: dict[str, int] = {}
    pairs: list[list[FactSpec]] = []
    if spec.scenario == "multifact":
        shuffled = chosen[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append(shuffled[i:i + 2])
        if len(shuffled) % 2:
            pairs.append([shuffled[-1]])
    else:
        pairs = [[f] for f in chosen]

    hi_plant = max(2, int(T * 0.55))
    positions = rng.sample(range(1, hi_plant), min(len(pairs), hi_plant - 1))
    positions.sort()
    for pos, group in zip(positions, pairs):
        if len(group) == 1:
            text = persona.style_plant(group[0].statement, rng)
        else:
            rest = "; ".join(g.statement[0].lower() + g.statement[1:].rstrip(".") for g in group[1:])
            text = persona.style_plant(f"{group[0].statement} Also, {rest}.", rng)
        slots[pos] = text
        occupied.add(pos)
        for g in group:
            plants[g.key] = pos

    # --- scenario: updates --------------------------------------------------
    updates: dict[str, int] = {}          # fact key → update turn
    stale_of: dict[str, list[str]] = {}   # fact key → superseded keywords
    if spec.scenario == "updates":
        upd_facts = [f for f in chosen if f.update][: spec.n_temporal]
        for i, f in enumerate(upd_facts):
            want = int(T * (0.5 + 0.12 * i))
            t = _free_near(slots, occupied, want, want - 5, want + 10)
            if t is None:
                continue
            slots[t] = persona.style_plant(f.update, rng)
            occupied.add(t)
            updates[f.key] = t
            stale_of[f.key] = [k.lower() for k in f.expected]

    # --- scenario: adversarial twins ----------------------------------------
    if spec.scenario == "adversarial" and topic.twins:
        a, b = topic.twins[0]
        if a.key in plants and b.key in plants:
            stale_of[a.key] = list(b.expected)
            stale_of[b.key] = list(a.expected)

    # --- scenario: distractors ----------------------------------------------
    if spec.scenario == "distractors":
        zone_hi = max(3, int(T * 0.7))
        for text in topic.distractors:
            t = _free_near(slots, occupied, rng.randint(2, zone_hi), 2, zone_hi)
            if t is not None:
                slots[t] = persona.style_plant(text, rng)
                occupied.add(t)

    # --- probes across lag zones ---------------------------------------------
    probes: list[Probe] = []

    def _stale_for(fact: FactSpec, t: int) -> list[str]:
        """Values whose mention indicates confusion at probe turn ``t``."""
        if spec.scenario == "adversarial" and fact.key in stale_of:
            return list(stale_of[fact.key])          # the twin's value
        if fact.key in updates and t > updates[fact.key]:
            return list(stale_of.get(fact.key, []))  # the superseded old value
        return []

    def add_probe(probe: Probe) -> None:
        occupied.add(probe.after_turn)
        probes.append(probe)

    kind_cycle = ("direct", "paraphrase", "multihop", "direct")
    ki = 0
    probe_turn_limit = T - 1
    for zone_name, zlo, zhi in LAG_ZONES:
        candidates = [f for f in chosen if f.key in plants]
        rng.shuffle(candidates)
        placed = 0
        for f in candidates * 3:
            if placed >= spec.probes_per_zone:
                break
            p = plants[f.key]
            lo = p + zlo
            hi = min(p + zhi, probe_turn_limit)
            if lo > hi:
                continue
            want = rng.randint(lo, hi)
            t = _free_near(slots, occupied, want, lo, hi)
            if t is None:
                continue
            kind = kind_cycle[ki % len(kind_cycle)]
            ki += 1
            if kind == "multihop":
                others = [g for g in chosen if g.key in plants and g.key != f.key and plants[g.key] < t]
                if not others:
                    kind = "direct"
                else:
                    g = rng.choice(others)
                    question = f"{f.question} And also, {g.paraphrase[0].lower()}{g.paraphrase[1:]}"
                    add_probe(Probe(after_turn=t, question=persona.style_question(question, rng),
                                    expected=[*f.expected, *g.expected],
                                    plant_turn=max(p, plants[g.key]), paraphrase=False,
                                    kind="multihop", stale=_stale_for(f, t)))
                    placed += 1
                    continue
            after_update = f.key in updates and t > updates[f.key]
            expected = list(f.new_expected) if (after_update and f.update) else list(f.expected)
            question = f.paraphrase if kind == "paraphrase" else f.question
            add_probe(Probe(after_turn=t, question=persona.style_question(question, rng),
                            expected=expected, plant_turn=p,
                            stale=_stale_for(f, t), paraphrase=(kind == "paraphrase"), kind=kind))
            placed += 1

    # --- temporal probes (updates scenario) ----------------------------------
    if spec.scenario == "updates":
        for f in [x for x in chosen if x.update and x.key in updates][: spec.n_temporal]:
            if not f.temporal_question:
                continue
            u = updates[f.key]
            t = _free_near(slots, occupied, u + rng.randint(4, 18), u + 2, min(T - 1, u + 25))
            if t is None:
                continue
            add_probe(Probe(after_turn=t, question=persona.style_question(f.temporal_question, rng),
                            expected=list(f.expected), plant_turn=plants[f.key],
                            stale=list(f.new_expected), paraphrase=False, kind="temporal"))

    # --- unanswerable probes (holdout facts) ----------------------------------
    hold = list(topic.holdout)
    rng.shuffle(hold)
    for h in hold[: spec.n_unanswerable]:
        lo = max(int(T * 0.55), 2)
        t = _free_near(slots, occupied, rng.randint(lo, T - 1), lo, T - 1)
        if t is None:
            continue
        add_probe(Probe(after_turn=t, question=persona.style_question(h.question, rng),
                        expected=[], forbidden=list(h.expected), kind="unanswerable"))

    # --- remaining turns: small talk ------------------------------------------
    filler_pool = list(topic.fillers) + list(GENERIC_FILLERS)
    for i in range(T):
        if slots[i] is None:
            if persona.name in ("terse", "sloppy"):
                slots[i] = rng.choice(list(persona.fillers) + filler_pool)
            else:
                slots[i] = rng.choice(filler_pool) if rng.random() < 0.5 else rng.choice(persona.fillers)

    return DialogSession(name=spec.spec_id, opening=[s for s in slots], fillers=[], probes=probes)


# ------------------------------------------------------------------ validation


def validate_topics(topics: tuple[Topic, ...] = TOPICS) -> list[str]:
    """Static pool checks: keyword uniqueness, required scenario material."""
    problems: list[str] = []
    for topic in topics:
        if len(topic.facts) < 8:
            problems.append(f"{topic.name}: fewer than 8 facts")
        updatable = [f for f in topic.facts if f.update]
        if len(updatable) < 2:
            problems.append(f"{topic.name}: fewer than 2 updatable facts")
        for f in updatable:
            if not f.temporal_question or not f.new_expected:
                problems.append(f"{topic.name}/{f.key}: update without temporal_question/new_expected")
        if not topic.twins:
            problems.append(f"{topic.name}: no twin pair for the adversarial scenario")
        if len(topic.holdout) < 2:
            problems.append(f"{topic.name}: fewer than 2 holdout facts")
        if len(topic.distractors) < 3:
            problems.append(f"{topic.name}: fewer than 3 distractors")

        owners: dict[str, str] = {}
        all_facts = [*topic.facts, *topic.holdout, *[x for pair in topic.twins for x in pair]]
        for f in all_facts:
            for kw in _fact_keywords(f):
                if kw in owners and owners[kw] != f.key:
                    problems.append(f"{topic.name}: keyword {kw!r} shared by {owners[kw]} and {f.key}")
                owners[kw] = f.key
            # expected keywords must literally occur in the corresponding statement
            for kw in f.expected:
                if kw.lower() not in f.statement.lower():
                    problems.append(f"{topic.name}/{f.key}: expected {kw!r} missing from statement")
            if f.update:
                for kw in f.new_expected:
                    if kw.lower() not in f.update.lower():
                        problems.append(f"{topic.name}/{f.key}: new expected {kw!r} missing from update")
        # cross-fact substring collisions ("may" vs "maybe", "anna" vs "hanna")
        keys = sorted(owners)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if owners[a] != owners[b] and (a in b or b in a):
                    problems.append(f"{topic.name}: colliding keywords {a!r} ({owners[a]}) / {b!r} ({owners[b]})")
        # no keyword may leak into distractors, fillers or the opener
        extra = [*topic.distractors, *topic.fillers, topic.label]
        for text in extra:
            low = text.lower()
            for kw, owner in owners.items():
                if kw in low:
                    problems.append(f"{topic.name}: keyword {kw!r} ({owner}) leaks into filler/distractor")
    return problems


def effective_turns(session: DialogSession) -> list[str]:
    """Turns as the model actually sees them (probes replace scripted lines)."""
    turns = list(session.user_turns())
    for p in session.probes:
        while len(turns) <= p.after_turn:
            turns.append("")
        turns[p.after_turn] = p.question
    return turns


def validate_session(session: DialogSession) -> list[str]:
    """Per-dialogue checks: ordering, lag zones, keyword presence/uniqueness."""
    problems: list[str] = []
    turns = effective_turns(session)
    seen_turns: set[int] = set()
    for p in session.probes:
        if p.after_turn in seen_turns:
            problems.append(f"{session.name}: duplicate probe turn {p.after_turn}")
        seen_turns.add(p.after_turn)
        if p.after_turn >= len(turns):
            problems.append(f"{session.name}: probe turn {p.after_turn} beyond dialogue")
            continue
        if p.plant_turn is not None:
            if p.after_turn <= p.plant_turn:
                problems.append(f"{session.name}: probe at {p.after_turn} not after plant {p.plant_turn}")
            lag = p.after_turn - p.plant_turn
            if not any(lo <= lag <= hi for _, lo, hi in LAG_ZONES):
                problems.append(f"{session.name}: probe lag {lag} outside all zones")
        if p.kind == "unanswerable":
            for kw in p.forbidden:
                for i, t in enumerate(turns):
                    if kw.lower() in t.lower():
                        problems.append(f"{session.name}: forbidden {kw!r} of an unanswerable probe appears at turn {i}")
                        break
        else:
            prefix = " ".join(turns[: p.after_turn]).lower()
            for kw in p.expected:
                if kw.lower() not in prefix:
                    problems.append(f"{session.name}: expected {kw!r} not planted before probe at turn {p.after_turn}")
            for kw in p.stale:
                if p.kind == "temporal" and kw.lower() not in prefix:
                    problems.append(f"{session.name}: temporal probe: new value {kw!r} not planted before turn {p.after_turn}")
    return problems


# ------------------------------------------------------------------ presets


def _combos(topics=TOPIC_NAMES, personas=PERSONA_NAMES, scenarios=SCENARIOS) -> list[tuple[str, str, str]]:
    """Round-robin combination order: any prefix covers all axes evenly.

    Step ``r`` yields ``(topics[r % 6], personas[r % 4], scenarios[r % 5])``;
    by CRT the first 60 combinations are unique and balanced — the first 12
    already touch every topic and every scenario.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    r = 0
    total = len(topics) * len(personas) * len(scenarios)
    while len(out) < total and r < total * 10:
        c = (topics[r % len(topics)], personas[r % len(personas)], scenarios[r % len(scenarios)])
        if c not in seen:
            seen.add(c)
            out.append(c)
        r += 1
    return out


def pilot_specs() -> list[CorpusSpec]:
    """12 dialogues: 2 topics × 2 personas × 3 scenarios, 100 turns."""
    combos = [(t, p, s) for t in TOPIC_NAMES[:2] for p in PERSONA_NAMES[:2] for s in SCENARIOS[:3]]
    return [
        CorpusSpec(spec_id=f"pilot-{i:02d}-{t}-{p}-{s}", topic=t, persona=p, scenario=s,
                   target_turns=100, seed=1000 + i)
        for i, (t, p, s) in enumerate(combos)
    ]


def main_specs() -> list[CorpusSpec]:
    """48 dialogues: first 24 balanced combos × lengths {100, 200}."""
    combos = _combos()[:24]
    specs: list[CorpusSpec] = []
    for li, length in enumerate((100, 200)):
        for i, (t, p, s) in enumerate(combos):
            specs.append(CorpusSpec(
                spec_id=f"main-L{length}-{i:02d}-{t}-{p}-{s}", topic=t, persona=p, scenario=s,
                target_turns=length, seed=5000 + 100 * li + i,
                probes_per_zone=5 if length >= 100 else 4,
            ))
    return specs


def ablation_specs() -> list[CorpusSpec]:
    """16 dialogues for the component-ablation ladder (100 turns)."""
    return [
        CorpusSpec(spec_id=f"abl-{i:02d}-{t}-{p}-{s}", topic=t, persona=p, scenario=s,
                   target_turns=100, seed=3000 + i)
        for i, (t, p, s) in enumerate(_combos()[:16])
    ]


def sweep_specs() -> list[CorpusSpec]:
    """8 dialogues for hyper-parameter sweeps (kv_slots, rag_k, budget)."""
    return [
        CorpusSpec(spec_id=f"sweep-{i:02d}-{t}-{p}-{s}", topic=t, persona=p, scenario=s,
                   target_turns=100, seed=7000 + i)
        for i, (t, p, s) in enumerate(_combos()[:8])
    ]


PRESETS = {
    "pilot": pilot_specs,
    "main": main_specs,
    "ablation": ablation_specs,
    "sweep": sweep_specs,
}


def specs_from_json(path) -> list[CorpusSpec]:
    """Load a custom spec list: JSON array of CorpusSpec dicts."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    return [CorpusSpec.from_dict(d) for d in data]
