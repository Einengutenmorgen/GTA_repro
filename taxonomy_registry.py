# taxonomy_registry.py
"""
Seeded-taxonomy registry for the taxonomy-seeded coding experiments (Aug 2026).

WHAT THIS IS FOR
----------------
The main pipeline builds the axial layer inductively (run_axial_coding). The
taxonomy-seeded experiments instead SKIP emergent axial coding and inject a
pre-existing human taxonomy "as if the first aggregation had already produced
it," then match the emergent open codes against that fixed taxonomy and study
where they refuse to fit (the edge cases). See the project design doc
GTA_taxonomy_seeded_experiment.md.

SemEval-2025 Task 10 supplies TWO structurally different taxonomies, seeded in
two INDEPENDENT experiments (each seeds only its FINE-grained layer; coarse
roll-up is a separately-scoped later experiment, left as a documented hook):

  seed="entity_role" (Subtask 1) : domain-AGNOSTIC. 3 main roles ->
      22 fine sub-roles. Native unit = an entity mention. This module hardcodes
      the 22 fine sub-roles from the official Subtask-1 taxonomy PDF
      (Description -> definition, Example -> anchors, main role -> parent).

  seed="narrative" (Subtask 2)   : domain-SPLIT (separate trees for the
      Ukraine-Russia War 'URW' and Climate Change 'CC' domains). Narrative ->
      Sub-narrative. Native unit = the whole article. load_taxonomy(...,
      domain=...) returns ONLY that domain's tree (domain-scoped seeding).

Subtask 3 (free-text narrative extraction) has NO taxonomy and is out of scope
for these experiments (noted as a future comparison against the final memo).

FIREWALL (load-bearing, mirrors article_chunking.py's labels/ hard-exclude)
---------------------------------------------------------------------------
Only the taxonomy SCHEMA (names, definitions, examples/anchors) may ever reach
a prompt/model -- that is the seeded axial layer, and showing it to the model
is the entire point of the experiment.

The GOLD per-instance ANSWERS (which entity got which role; which sub-narrative
labels an article carries) must NEVER reach a prompt. They are evaluation-only.
load_gold() is the ONLY entry point to them, guarded by _GOLD_IS_EVAL_ONLY and
a loud contract: its return value must be passed ONLY to
taxonomy_match.score_against_gold, never to any prompt builder. A grep-based
firewall test (test_taxonomy_seeded.py) enforces this by asserting no rendered
prompt ever contains a gold label token.

Dependencies: stdlib only (os, glob, re). No numpy / no model here.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Datasets this registry knows about (kept in sync with prompt_registry.DATASETS
# conceptually, but imported lazily to avoid a hard import cycle at module load).
KNOWN_DATASETS = ("semeval", "silan")
KNOWN_SEEDS = ("entity_role", "narrative")
KNOWN_LEVELS = ("fine", "coarse")
KNOWN_DOMAINS = ("URW", "CC")

# EVAL-ONLY guard. load_gold()'s output is scored against, never shown to the
# model. This constant exists so callers/tests can assert intent explicitly.
_GOLD_IS_EVAL_ONLY = True


# ---------------------------------------------------------------------------
# Common taxonomy schema (identical shape whichever seed / dataset it came from,
# so taxonomy_match.py and the seeded-axial injection are seed-agnostic).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaxonomyLeaf:
    """One fine-grained category the open codes are matched against.

    category_id : stable machine id (e.g. "antagonist/saboteur", "URW/discrediting_ukraine/rewriting_history").
    name        : human display name ("Saboteur", "Rewriting Ukraine's history").
    definition  : the taxonomy's Description/Definition prose for this leaf.
    anchors     : example strings (the taxonomy's Example / Instructions-to-Annotators).
                  Concatenated with `definition` to form the embedding index text
                  and the {taxonomy_block} shown to the LLM matcher.
    parent      : coarse-parent id ("antagonist", "URW/discrediting_ukraine").
                  Retained so the DEFERRED coarse roll-up experiment (E3) can
                  test fine->coarse consistency without reloading anything.
    level       : "fine" (always, for the built experiments).
    domain      : "URW" | "CC" for narrative leaves; None for domain-agnostic
                  entity-role leaves.
    """
    category_id: str
    name: str
    definition: str
    anchors: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    level: str = "fine"
    domain: Optional[str] = None

    def index_text(self) -> str:
        """Text embedded to represent this leaf in embed_match (definition +
        anchors). Deterministic, so the embedding disk-cache key is stable."""
        parts = [self.name, self.definition] + list(self.anchors)
        return " \n ".join(p for p in parts if p)


@dataclass(frozen=True)
class Taxonomy:
    """A seeded taxonomy: the fixed axial layer for one experiment."""
    dataset: str
    seed: str
    level: str
    domain: Optional[str]
    leaves: List[TaxonomyLeaf]

    def as_axial_relations(self) -> List[dict]:
        """Render the taxonomy in the SAME structural shape run_axial_coding
        emits (a list of category dicts), so everything downstream is
        format-identical whether the axial layer was emergent or seeded. The
        paradigm slots are left empty here -- this is a SEED, not a resolved
        category; slot escalation is a different experiment (§14) and does not
        run on the seeded path."""
        return [
            {
                "axial_category": lf.name,
                "category_id": lf.category_id,
                "reasoning": f"SEEDED from {self.seed} taxonomy leaf; not emergent.",
                "definition": lf.definition,
                "anchors": lf.anchors,
                "parent": lf.parent,
                "domain": lf.domain,
                "__seeded__": True,
            }
            for lf in self.leaves
        ]

    def taxonomy_block(self) -> str:
        """The rendered category list shown to the LLM matcher as
        {taxonomy_block}. Schema only -- never any gold assignment."""
        lines = []
        for lf in self.leaves:
            head = f"- {lf.name}"
            if lf.parent:
                head += f"  (parent: {lf.parent})"
            lines.append(head)
            lines.append(f"    definition: {lf.definition}")
            if lf.anchors:
                lines.append(f"    examples: {' | '.join(lf.anchors)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SEED A -- Entity-role taxonomy (Subtask 1). Domain-agnostic. 22 fine sub-roles.
# Source: SemEval-2025 Task 10 Subtask-1 "Entity Framing and Role Portrayal"
# taxonomy PDF (Description -> definition, Example -> anchors, main role ->
# parent). Verbatim-faithful paraphrase of the PDF's Detailed Taxonomy section.
# ---------------------------------------------------------------------------

_ENTITY_ROLE_FINE: List[TaxonomyLeaf] = [
    # ---- Protagonist ----
    TaxonomyLeaf("protagonist/guardian", "Guardian",
        "Heroes or guardians who protect values or communities, ensuring safety and "
        "upholding justice. They often take on roles such as law enforcement officers, "
        "soldiers, or community leaders.",
        ["Police officers protecting citizens during a crisis; firefighters saving lives; "
         "community leaders standing up against crime or for action to address climate change."],
        parent="protagonist"),
    TaxonomyLeaf("protagonist/martyr", "Martyr",
        "Martyrs or saviors who sacrifice their well-being, or even their lives, for a "
        "greater good or cause. Celebrated for selflessness and dedication. Mostly in "
        "politics, not in climate change.",
        ["Civil rights leaders like Martin Luther King Jr.; journalists who risk their "
         "lives to report on corruption and injustice."],
        parent="protagonist"),
    TaxonomyLeaf("protagonist/peacemaker", "Peacemaker",
        "Individuals who advocate for harmony, working tirelessly to resolve conflicts and "
        "bring about peace, often through diplomacy, negotiation, and mediation. Mostly in "
        "politics, not in climate change.",
        ["Nelson Mandela's efforts to reconcile South Africa post-apartheid; diplomats "
         "brokering peace deals between conflicting nations."],
        parent="protagonist"),
    TaxonomyLeaf("protagonist/rebel", "Rebel",
        "Rebels, revolutionaries, or freedom fighters who challenge the status quo and "
        "fight for significant change or liberation from oppression; seen as champions of "
        "justice and freedom.",
        ["Independence-movement leaders like Mahatma Gandhi; modern activists fighting for "
         "democratic reforms; in the CC domain, figures like Greta Thunberg, or persons who "
         "chain themselves to trees to prevent deforestation."],
        parent="protagonist"),
    TaxonomyLeaf("protagonist/underdog", "Underdog",
        "Entities considered unlikely to succeed due to their disadvantaged position but "
        "who strive against greater forces and obstacles; their stories often inspire.",
        ["Grassroots political candidates overcoming well-funded incumbents; small nations "
         "standing up to more powerful countries; underfunded organizations framed as making "
         "a positive impact on CC."],
        parent="protagonist"),
    TaxonomyLeaf("protagonist/virtuous", "Virtuous",
        "Individuals portrayed as virtuous, righteous, or noble -- fair, just, and upholding "
        "high moral standards; role models and figures of integrity.",
        ["Judges known for fairness; politicians with a reputation for honesty; leaders "
         "standing up for environmental ethical values, or activists pushing for "
         "environmental sustainability."],
        parent="protagonist"),
    # ---- Antagonist ----
    TaxonomyLeaf("antagonist/instigator", "Instigator",
        "Individuals or groups initiating conflict, often seen as the primary cause of "
        "tension and discord; they may provoke violence or unrest.",
        ["Politicians using inflammatory rhetoric to incite violence; groups instigating "
         "protests to destabilize governments; in CC, activists framed negatively as "
         "troublemakers (who might then also take the Saboteur sub-role)."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/conspirator", "Conspirator",
        "Those involved in plots and secret plans, often working behind the scenes to "
        "undermine or deceive others; they engage in covert activities.",
        ["Figures in political scandals or espionage, such as Watergate conspirators or cyber "
         "espionage cases; in CC, persons/organizations conspiring to bypass environmental "
         "regulations for profit."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/tyrant", "Tyrant",
        "Tyrants and corrupt officials who abuse their power, ruling unjustly and oppressing "
        "those under their control; characterized by authoritarian rule and exploitation.",
        ["Dictators like Kim Jong-un; corrupt officials embezzling public funds and "
         "suppressing dissent."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/foreign_adversary", "Foreign Adversary",
        "Entities from other nations or regions creating geopolitical tension and acting "
        "against the interests of another country; often depicted as threats to national "
        "security. Mostly in politics, not in climate change.",
        ["Rival nations in espionage or military confrontations, e.g. Cold War adversaries; "
         "in CC, portrayals of other countries not adhering to CC policies (e.g. a country "
         "refusing to cut CO2 emissions)."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/traitor", "Traitor",
        "Individuals who betray a cause or country, seen as disloyal and treacherous; their "
        "actions are a significant breach of trust. Mostly in politics, not in CC.",
        ["Whistleblowers revealing sensitive information for personal gain; soldiers "
         "defecting to enemy forces. (If a whistleblower is portrayed positively, the role "
         "would instead be Virtuous.)"],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/spy", "Spy",
        "Spies or double agents accused of espionage, gathering and transmitting sensitive "
        "information to a rival or enemy; they operate in secrecy and deception. Mostly in "
        "politics, not in CC.",
        ["Historical figures like Aldrich Ames, who spied for the Soviet Union; contemporary "
         "corporate espionage."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/saboteur", "Saboteur",
        "Saboteurs who deliberately damage or obstruct systems, processes, or organizations "
        "to cause disruption or failure; they aim to weaken or destroy targets from within.",
        ["Insiders tampering with critical infrastructure; activists sabotaging industrial "
         "operations."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/corrupt", "Corrupt",
        "Individuals or entities that engage in unethical or illegal activities for personal "
        "gain, prioritizing profit or power over ethics; includes corrupt politicians, "
        "business leaders, and officials.",
        ["Companies involved in environmental pollution; executives engaged in massive "
         "financial fraud; politicians accepting bribes and engaging in graft."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/incompetent", "Incompetent",
        "Entities causing harm through ignorance, lack of skill, or incompetence -- foolish "
        "acts or poor decisions due to lack of understanding or expertise, often "
        "unintentional but with significant negative consequences.",
        ["Leaders making reckless policy decisions without proper understanding; officials "
         "mishandling crisis responses; managers whose poor judgment leads to organizational "
         "failures."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/terrorist", "Terrorist",
        "Terrorists, mercenaries, insurgents, fanatics, or extremists engaging in violence "
        "and terror to further ideological ends, often targeting civilians; viewed as "
        "significant threats to peace and security. Mostly in politics, not in CC.",
        ["Groups like ISIS or Al-Qaeda carrying out attacks; lone-wolf terrorists committing "
         "acts of violence."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/deceiver", "Deceiver",
        "Deceivers, manipulators, or propagandists who twist the truth, spread "
        "misinformation, and manipulate public perception for their own benefit; they "
        "undermine trust and truth.",
        ["Politicians spreading false information for political gain; media outlets engaging "
         "in propaganda."],
        parent="antagonist"),
    TaxonomyLeaf("antagonist/bigot", "Bigot",
        "Individuals accused of hostility or discrimination against specific groups -- acts "
        "of racism, sexism, homophobia, antisemitism, Islamophobia, or any kind of hate "
        "speech. Mostly in politics, not in CC.",
        ["Entities committing or endorsing hate speech or discrimination against a group."],
        parent="antagonist"),
    # ---- Innocent ----
    TaxonomyLeaf("innocent/forgotten", "Forgotten",
        "Marginalized or overlooked groups who are often ignored by society and do not "
        "receive the attention or support they need; includes refugees facing systemic "
        "neglect and exclusion.",
        ["Indigenous populations facing ongoing discrimination; homeless individuals "
         "struggling without adequate support; refugees fleeing conflict or persecution."],
        parent="innocent"),
    TaxonomyLeaf("innocent/exploited", "Exploited",
        "Individuals or groups used for others' gain, often without their consent and with "
        "significant detriment to their well-being; often victims of labor exploitation, "
        "trafficking, or economic manipulation.",
        ["Workers in sweatshops; victims of human trafficking; communities suffering from "
         "corporate exploitation of natural resources."],
        parent="innocent"),
    TaxonomyLeaf("innocent/victim", "Victim",
        "People cast as victims due to circumstances beyond their control, in two categories: "
        "(1) victims of physical harm (natural disasters, acts of war, terrorism, mugging, "
        "physical assault, etc.), and (2) victims of economic harm (sanctions, blockades, "
        "boycotts). Their experiences evoke sympathy and calls for justice.",
        ["Victims of natural disasters such as hurricanes or earthquakes; individuals "
         "affected by violent crimes; victims of economic blockades, sanctions, or boycotts."],
        parent="innocent"),
    TaxonomyLeaf("innocent/scapegoat", "Scapegoat",
        "Entities blamed unjustly for problems or failures, often to divert attention from "
        "the real causes or culprits; made to bear the brunt of criticism and punishment "
        "without just cause.",
        ["Minority groups blamed for economic problems; political opponents accused of "
         "provoking national strife without evidence."],
        parent="innocent"),
]


# ---------------------------------------------------------------------------
# SEED B -- Narrative taxonomy (Subtask 2). Domain-SPLIT (URW / CC), two-level.
# Source: SemEval-2025 Task 10 Subtask-2 "Narrative Classification" taxonomy
# PDFs (Definition -> definition; Instructions-to-Annotators + Example ->
# anchors; parent Narrative -> parent). The FINE layer is the sub-narratives.
#
# NOTE: the full sub-narrative definitions run to many pages; this registry
# hardcodes the label hierarchy (Narrative -> Sub-narrative) verbatim from the
# taxonomy figures, with the sub-narrative DEFINITIONS filled in for the nodes
# whose definition text is in the provided PDF pages and a definition==name
# fallback elsewhere. Fill remaining definitions from the Subtask-2 definitions
# PDF before a real run (they only strengthen the embedding index; the label
# hierarchy itself is complete and correct here). This is the single place to
# paste them -- taxonomy_match.py never hardcodes labels.
# ---------------------------------------------------------------------------

def _narr(parent_name: str, sub_pairs, domain: str) -> List[TaxonomyLeaf]:
    """Helper: build fine sub-narrative leaves under one parent narrative.
    sub_pairs is a list of (sub_name, definition_or_None)."""
    pslug = _slug(parent_name)
    out = []
    for sub_name, defn in sub_pairs:
        out.append(TaxonomyLeaf(
            category_id=f"{domain}/{pslug}/{_slug(sub_name)}",
            name=sub_name,
            definition=defn or sub_name,  # fallback: label as its own definition
            anchors=[],
            parent=f"{domain}/{pslug}",
            domain=domain,
        ))
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


# --- URW (Ukraine-Russia War) fine sub-narratives -------------------------
_URW_NARRATIVE_FINE: List[TaxonomyLeaf] = (
    _narr("Blaming the war on others rather than the invader", [
        ("Ukraine is the aggressor",
         "Statements that shift the responsibility of the aggression to Ukraine instead of "
         "Russia and portray Ukraine as the attacker."),
        ("The West are the aggressors",
         "Statements that shift the responsibility for the conflict and escalation to the "
         "Western block."),
    ], "URW") +
    _narr("Discrediting Ukraine", [
        ("Rewriting Ukraine's history",
         "Statements that aim to reestablish history of the Ukrainian nation in a way that "
         "discredits its reputation."),
        ("Discrediting Ukrainian nation and society",
         "Statements that aggressively undermine the legitimacy and reputability of Ukrainian "
         "ethnicity and people."),
        ("Discrediting Ukrainian military",
         "Statements that aim to undermine the capabilities, professionalism or effectiveness "
         "of the Ukrainian armed forces."),
        ("Discrediting Ukrainian government and officials and policies",
         "Statements that seek to delegitimize the Ukrainian government, its leaders, and its "
         "policies, portraying them as corrupt or incompetent."),
        ("Ukraine is a puppet of the West",
         "Claims that Ukraine is controlled or heavily influenced by Western powers, "
         "particularly the United States and European Union."),
        ("Ukraine is a hub for criminal activities",
         "Allegations that Ukraine is a center for illegal activities such as human "
         "trafficking, drug smuggling, or organized crime."),
        ("Ukraine is associated with nazism",
         "Accusations that Ukrainian society or government has ties to or sympathies with "
         "Nazi ideology, often referencing historical events or extremist groups."),
        ("Situation in Ukraine is hopeless",
         "Statements that portray Ukraine as having no viable perspectives or no potential "
         "positive future."),
    ], "URW") +
    _narr("Russia is the Victim", [
        ("The West is russophobic",
         "Statements that claim the negative reaction to Russia's actions are because of the "
         "negative perspective of western countries instead of Russia's own actions."),
        ("Russia actions in Ukraine are only self-defence",
         "Statements that justify Russia's action solely as legitimate self-defence and not a "
         "deliberate action."),
        ("UA is anti-RU extremists",
         "Statements claiming that Ukraine is comprised of extremist elements that are "
         "vehemently opposed to Russia."),
    ], "URW") +
    _narr("Praise of Russia", [
        ("Praise of Russian military might",
         "Statements that positively highlight Russia's military institutions, equipment and "
         "scale."),
        ("Praise of Russian President Vladimir Putin",
         "Statements that present Vladimir Putin positively, including his personal and "
         "leadership qualities."),
        ("Russia is a guarantor of peace and prosperity",
         "Statements that portray Russia solely in a positive manner, emphasising its "
         "potential to provide peace and prosperity to those that cooperate."),
        ("Russia has international support from a number of countries and people", None),
        ("Russian invasion has strong national support", None),
    ], "URW") +
    _narr("Overpraising the West", [
        ("NATO will destroy Russia", None),
        ("The West belongs in the right side of history", None),
        ("The West has the strongest international support", None),
    ], "URW") +
    _narr("Speculating war outcomes", [
        ("Russian army is collapsing", None),
        ("Russian army will lose all the occupied territories", None),
        ("Ukrainian army is collapsing", None),
    ], "URW") +
    _narr("Discrediting the West, Diplomacy", [
        ("The EU is divided", None),
        ("The West is weak", None),
        ("The West is overreacting", None),
        ("The West does not care about Ukraine, only about its interests", None),
        ("Diplomacy does/will not work", None),
        ("West is tired of Ukraine", None),
    ], "URW") +
    _narr("Negative Consequences for the West", [
        ("Sanctions imposed by Western countries will backfire", None),
        ("The conflict will increase the Ukrainian refugee flows to Europe", None),
    ], "URW") +
    _narr("Distrust towards Media", [
        ("Western media is an instrument of propaganda", None),
        ("Ukrainian media cannot be trusted", None),
    ], "URW") +
    _narr("Amplifying war-related fears", [
        ("By continuing the war we risk WWIII", None),
        ("Russia will also attack other countries", None),
        ("There is a real possibility that nuclear weapons will be employed", None),
        ("NATO should/will directly intervene", None),
    ], "URW") +
    _narr("Hidden plots by secret schemes of powerful groups", [
        # This narrative appears with no enumerated sub-narratives in the figure;
        # seed a single self-named leaf so the parent is still matchable.
        ("Hidden plots by secret schemes of powerful groups", None),
    ], "URW")
)


# --- CC (Climate Change) fine sub-narratives ------------------------------
_CC_NARRATIVE_FINE: List[TaxonomyLeaf] = (
    _narr("Criticism of climate policies", [
        ("Climate policies are ineffective", None),
        ("Climate policies have negative impact on the economy", None),
        ("Climate policies are only for profit", None),
    ], "CC") +
    _narr("Criticism of institutions and authorities", [
        ("Criticism of the EU", None),
        ("Criticism of international entities", None),
        ("Criticism of national governments", None),
        ("Criticism of political organizations and figures", None),
    ], "CC") +
    _narr("Climate change is beneficial", [
        ("CO2 is beneficial", None),
        ("Temperature increase is beneficial", None),
    ], "CC") +
    _narr("Downplaying climate change", [
        ("Climate cycles are natural", None),
        ("Weather suggests the trend is global cooling", None),
        ("Temperature increase does not have significant impact", None),
        ("CO2 concentrations are too small to have an impact", None),
        ("Human activities do not impact climate change", None),
        ("Ice is not melting", None),
        ("Sea levels are not rising", None),
        ("Humans and nature will adapt to the changes", None),
    ], "CC") +
    _narr("Questioning the measurements and science", [
        ("Methodologies/metrics used are unreliable/faulty", None),
        ("Data shows no temperature increase", None),
        ("Greenhouse effect/carbon dioxide do not drive climate change", None),
        ("Scientific community is unreliable", None),
    ], "CC") +
    _narr("Criticism of climate movement", [
        ("Climate movement is alarmist", None),
        ("Climate movement is corrupt", None),
        ("Ad hominem attacks on key activists", None),
    ], "CC") +
    _narr("Controversy about green technologies", [
        ("Renewable energy is dangerous", None),
        ("Renewable energy is unreliable", None),
        ("Renewable energy is costly", None),
        ("Nuclear energy is not climate friendly", None),
    ], "CC") +
    _narr("Hidden plots by secret schemes of powerful groups", [
        ("Blaming global elites", None),
        ("Climate agenda has hidden motives", None),
    ], "CC") +
    _narr("Amplifying Climate Fears", [
        ("Earth will be uninhabitable soon", None),
        ("Amplifying existing fears of global warming", None),
        ("Doomsday scenarios for humans", None),
        ("Whatever we do it is already too late", None),
    ], "CC") +
    _narr("Green policies are geopolitical instruments", [
        ("Climate-related international relations are abusive/exploitative", None),
        ("Green activities are a form of neo-colonialism", None),
    ], "CC")
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def domain_of(source_id: str) -> Optional[str]:
    """Domain of a SemEval article, parsed from its filename-derived source_id.

    Filenames look like EN_CC_200081 (Climate) or EN_UA_DEV_100013 (Ukraine
    war). The domain token is CC / UA anywhere in the underscore-split id
    (case-insensitive). Returns "CC" | "URW" | None (None -> caller decides;
    for E2 that means the article can't be domain-scoped and should be flagged).
    """
    toks = {t.upper() for t in re.split(r"[^A-Za-z]+", source_id) if t}
    if "CC" in toks:
        return "CC"
    if "UA" in toks or "URW" in toks or "RW" in toks:
        return "URW"
    return None


def load_taxonomy(
    dataset: str,
    seed: str,
    level: str = "fine",
    domain: Optional[str] = None,
) -> Taxonomy:
    """Load a seeded taxonomy (the fixed axial layer) for one experiment.

    dataset : "semeval" (built) | "silan" (placeholder -> NotImplementedError).
    seed    : "entity_role" (domain-agnostic) | "narrative" (domain-scoped).
    level   : "fine" (built). "coarse" is the DEFERRED roll-up experiment (E3)
              and raises NotImplementedError with a pointer.
    domain  : REQUIRED for seed="narrative" ("URW" | "CC"); ignored (and must be
              None) for seed="entity_role".
    """
    ds = dataset.strip().lower()
    sd = seed.strip().lower()
    lv = level.strip().lower()

    if sd not in KNOWN_SEEDS:
        raise ValueError(f"unknown seed {seed!r}; expected one of {KNOWN_SEEDS}")
    if lv not in KNOWN_LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {KNOWN_LEVELS}")

    if ds == "silan":
        # PLACEHOLDER (documented hook, deliberately not built now). The Silan
        # per-country codebook xlsx already carries Theme->Code->Definition->
        # Example columns; map Code -> fine TaxonomyLeaf (definition from
        # Definition col, anchors from Example col, parent = Theme). Two natural
        # levels: fine = Code, coarse = Theme -> Broadly-Shared-Theme (the
        # `Themes` sheet). Silan has NO gold answer key (its human codebook is
        # explicitly not ground truth), so there is no load_gold analogue.
        raise NotImplementedError(
            "Silan seeded-taxonomy is a documented placeholder: wire the "
            "per-country xlsx Theme->Code->Definition->Example columns into "
            "TaxonomyLeaf here (fine=Code, coarse=Theme). See "
            "GTA_taxonomy_seeded_experiment.md §11."
        )
    if ds != "semeval":
        raise ValueError(f"unknown dataset {dataset!r}; expected 'semeval' or 'silan'")

    if lv == "coarse":
        raise NotImplementedError(
            "Coarse roll-up (Experiment E3) is deferred. The fine leaves already "
            "carry `parent`, so E3 can roll up fine matches to the 3 main roles / "
            "top narratives and test fine->coarse consistency without new loading. "
            "See GTA_taxonomy_seeded_experiment.md §3."
        )

    if sd == "entity_role":
        if domain is not None:
            raise ValueError(
                "seed='entity_role' is domain-agnostic; pass domain=None "
                "(the same 22 sub-roles apply to both SemEval domains)."
            )
        return Taxonomy("semeval", "entity_role", "fine", None, list(_ENTITY_ROLE_FINE))

    # seed == "narrative": domain-scoped
    if domain is None:
        raise ValueError(
            "seed='narrative' is domain-split; pass domain='URW' or domain='CC'. "
            "Use domain_of(source_id) to resolve an article's domain from its filename."
        )
    dom = domain.strip().upper()
    if dom == "URW":
        leaves = list(_URW_NARRATIVE_FINE)
    elif dom == "CC":
        leaves = list(_CC_NARRATIVE_FINE)
    else:
        raise ValueError(f"unknown narrative domain {domain!r}; expected 'URW' or 'CC'")
    return Taxonomy("semeval", "narrative", "fine", dom, leaves)


# ---------------------------------------------------------------------------
# GOLD (EVAL-ONLY). NEVER pass any return value here into a prompt builder.
# ---------------------------------------------------------------------------

def load_gold(dataset: str, seed: str, labels_dir: str) -> dict:
    """Load the SemEval GOLD answer key for scoring only.

    *** EVAL-ONLY. The return value must be passed ONLY to
    taxonomy_match.score_against_gold, NEVER to any prompt/model. ***

    dataset : "semeval". "silan" has no gold key -> ValueError.
    seed    : "narrative" -> {source_id: set(subnarrative_names)} from the
              Subtask-2 annotations. "entity_role" -> {source_id:
              [(entity_mention, main_role, [fine_roles...]), ...]} from the
              Subtask-1 annotations.
    labels_dir : path to the SemEval labels/<LANG> directory (the hard-excluded
              tree). Caller must pass this EXPLICITLY -- it is never derived
              inside the pipeline arm, so gold-loading can only happen from the
              scorer, by design.

    Parses the official tab-separated annotation format:
      subtask-1: article_id <TAB> entity_mention <TAB> start <TAB> end <TAB> main_role <TAB> fine_role[,fine_role...]
      subtask-2: article_id <TAB> narratives <TAB> subnarratives   (';'-separated)
    Tolerant to minor column-count variation; unknown/blank rows are skipped.
    """
    assert _GOLD_IS_EVAL_ONLY, "gold is eval-only"
    ds = dataset.strip().lower()
    sd = seed.strip().lower()
    if ds == "silan":
        raise ValueError(
            "Silan has no gold answer key (its human codebook is explicitly not "
            "ground truth); seeded matching yields soft edge-typing only."
        )
    if ds != "semeval":
        raise ValueError(f"unknown dataset {dataset!r}")

    def _iter_annotation_lines():
        pat = "subtask-1-*" if sd == "entity_role" else "subtask-2-*"
        for path in sorted(glob.glob(os.path.join(labels_dir, "**", pat), recursive=True)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.strip():
                        yield line

    def _sid(article_id: str) -> str:
        # Gold uses the article filename (with or without .txt); chunk source_id
        # is the stem. Normalize both to the stem.
        return os.path.splitext(article_id.strip())[0]

    if sd == "narrative":
        gold: Dict[str, set] = {}
        for line in _iter_annotation_lines():
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            sid = _sid(cols[0])
            subs = cols[2] if len(cols) >= 3 else ""
            labels = {s.strip() for s in re.split(r"[;|]", subs) if s.strip()
                      and s.strip().lower() != "other"}
            gold.setdefault(sid, set()).update(labels)
        return gold

    # entity_role
    gold_er: Dict[str, list] = {}
    for line in _iter_annotation_lines():
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        sid = _sid(cols[0])
        entity = cols[1].strip()
        main_role = cols[4].strip()
        fine = []
        if len(cols) >= 6:
            fine = [c.strip() for c in re.split(r"[,;|]", cols[5]) if c.strip()]
        gold_er.setdefault(sid, []).append((entity, main_role, fine))
    return gold_er
