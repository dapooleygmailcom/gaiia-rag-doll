# ASL Query Benchmark Results

## Case 1 [direct_rule]
**Query**: What does rule A7.212 say about entry of forces into the mapboard?

**Expected Keywords**: enter, mapboard, turn, movement
**Expected Rule**: A7.212

### Agent Answer (395.0s)
The provided rules do not contain information about what rule A7.212 says regarding entry of forces into the mapboard. The question cannot be answered using only the given text. [No direct reference]

**Debug**: Retrieved=8, XRefs=1
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.5
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.6
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 7.21

---

## Case 2 [direct_rule]
**Query**: What is the basic infantry movement rule A4.1?

**Expected Keywords**: attacker, move, infantry, fire, broken
**Expected Rule**: A4.1

### Agent Answer (489.1s)
The basic infantry movement rule A4.1 states that during their MPh, an ATTACKER may move all, some, or none of his Infantry units provided they did not fire during the PFPh and are neither broken, TI, designated to use Opportunity Fire, nor held in Melee [Rule 4.1].

**Debug**: Retrieved=8, XRefs=4
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.1
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.61
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 3.723

---

## Case 3 [direct_rule]
**Query**: What does rule D5.6 say about PRC survival checks?

**Expected Keywords**: prc, survival, eliminated, vehicle
**Expected Rule**: D5.6

### Agent Answer (642.3s)
Rule D5.6 does not provide specific details about PRC survival checks. For PRC survival checks, refer to rules related to PRC handling such as D5.31 [D5.31]. If rule D5.6 is meant in context of PRC survival and no direct statement exists within provided texts, it indicates that D5.6 may address different aspects or there might be reliance on other specific rules for PRCs' survival checks rather than itself stating them. Provided text does not contain the exact information regarding what rule D5.6 says about PRC survival checks [as per given data].

**Debug**: Retrieved=8, XRefs=3
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 9.2
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 5.6
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 6.9

---

## Case 4 [errata_supersession]
**Query**: What is the current definition of 'Armed' according to the latest errata?

**Expected Keywords**: armed, gun, functioning, sw
**Expected Rule**: None

### Agent Answer (665.2s)
According to the latest errata, the definition of 'Armed' is superseded by the following update:

"A unit is considered Armed if it possesses a weapon or equipment beyond just its inherent capabilities [A20.54]. The term now explicitly includes units with any additional weaponry or gear that goes beyond what is naturally provided, such as an inherent driver for vehicles."

However, the precise and updated definition from errata directly states:
"A unit is considered 'Armed' if it possesses a weapon or equipment beyond just its inherent capabilities. For example, a vehicle with only its inherent driver is 'Unarmed'. [Errata to A20.54]"

**Debug**: Retrieved=8, XRefs=7
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.22
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 20.551
- ASLRB_Errata_Dec_2025.pdf (P3) Rule: A15.5

---

## Case 5 [errata_supersession]
**Query**: What is the correct definition of 'Squad Equivalent' after the latest errata?

**Expected Keywords**: squad, equivalent, non-inherent, crew
**Expected Rule**: None

### Agent Answer (257.2s)
According to Rule 5.5 [Doc: core_rules], Five SMC equal a HS, and two crews/HS equal a squad, but four SMC count as zero squad-equivalents. However, the latest errata updates this rule by replacing "squads" with "squad-equivalents," meaning that two crews or HS are equivalent to one squad-equivalent [Doc: errata | Section: C13.311]. Additionally, if an Infantry crew/HS is manning a Gun, it counts as a squad for stacking purposes [Doc: core_rules | Rule: 5.5].

**Debug**: Retrieved=8, XRefs=0
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 5.5
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 1.121
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 1.122

---

## Case 6 [errata_supersession]
**Query**: What does the latest Field Phone definition say, after corrections?

**Expected Keywords**: field, phone, ocg6
**Expected Rule**: None

### Agent Answer (680.4s)
According to the errata rules provided:

The latest definition of Field Phone after corrections states that a MMC can attempt to Repair a SW and Scrounge a vehicle during the same RPh. However, the QA clarifies this point in contradiction by stating "A9.72 & D10.5 Can a MMC attempt to Repair a SW and Scrounge a vehicle in the same RPh? A. No; see A3.1." [A9.72]

Therefore, despite any initial impression given by other rules, according to QA clarification, a unit cannot both repair a small weapon (SW) and scrounge a vehicle during the same Rally Phase. This QA response supersedes any conflicting base rule definitions.

**Debug**: Retrieved=8, XRefs=4
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 1.23
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 1.22
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 1.2

---

## Case 7 [cross_reference]
**Query**: When a unit cannot enter a designated hex due to enemy occupation, what rules govern alternate entry?

**Expected Keywords**: A4.14, entry, blocked, alternate
**Expected Rule**: A7.212

### Agent Answer (474.7s)
When a unit is scheduled to enter a hex but finds it occupied by an enemy squad during its movement phase (MPh), Rule 2.5 [Doc: core_rules] states that entry must be made in a non-obstructed hex within four hexes of the original entry point, delayed until one Game Turn later if necessary. If no such hex is available, the radius for permissible entry extends another four hexes and may incur further delays until an unobstructed entry hex is found.

There are no specific rules provided that govern alternate entry directly in relation to enemy occupation beyond this initial rule. Therefore, the player must adhere to Rule 2.5 [Doc: core_rules] for managing forced entry delays and alternative entry points when encountering occupied locations.

**Debug**: Retrieved=8, XRefs=0
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 23.922
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 6.6
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.5

---

## Case 8 [cross_reference]
**Query**: What is a Minimum Move and when can it be claimed? Include the terrain cost exception.

**Expected Keywords**: minimum, move, mf, na, not allowed
**Expected Rule**: A4.134

### Agent Answer (714.6s)
A Minimum Move allows a unit to move up to 1 hex when it lacks sufficient Movement Factors (MF) or Movement Points (MP). [4.U]

For Personnel units:
- A unit can make a Minimum Move if it has at least one MF remaining after all other movement costs have been paid. [B3.4]
- Cavalry units may also perform a Minimum Move like Infantry, but both the Cavalry and its horse become Coherent eXhausted (CX) as an exception. [13.3]

For vehicular counters:
- A unit must have at least one MP remaining after paying all movement costs to claim a Minimum Move.

Terrain cost exceptions include when pushing or unhooking Guns; units cannot attempt repairs nor change the Gun's CA while attempting these actions, except as part of the hooking/unhooking procedure. [C10.1]

The base rules do not provide further specific terrain-related exceptions for performing a Minimum Move beyond gun handling operations.

**Debug**: Retrieved=8, XRefs=6
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.134
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.15
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 3.7331

---

## Case 9 [scenario]
**Query**: What are the victory conditions and special rules for Scenario A: The Guards Counterattack?

**Expected Keywords**: stalingrad, german, russian, victory, buildings
**Expected Rule**: None

### Agent Answer (630.4s)
The provided information does not contain specific details about Scenario A: The Guards Counterattack regarding victory conditions or special rules. Therefore, I am unable to provide those specifics based on the given data.

However, there is a cross-reference to Rule [A10.7] which might be relevant if this rule pertains directly to Scenario A's passenger fire modifications according to errata [ASL_HASL_Errata_Mar_2025.pdf], but no other details about victory conditions or special rules for Scenario A are provided in the text.

For precise and complete information on Scenario A: The Guards Counterattack, please refer to the full ASL rulebook or scenario booklet.

**Debug**: Retrieved=8, XRefs=7
**Top Sources**:
- pdfcoffee.com_wargame-avalon-hill-advanced-squad-leader-scenarios-1-to-74-pdf-free.pdf (P8) Rule: 
- pdfcoffee.com_wargame-avalon-hill-advanced-squad-leader-scenarios-1-to-74-pdf-free.pdf (P8) Rule: 
- pp4.pdf (P8) Rule: B25.13

---

## Case 10 [scenario]
**Query**: What are the forces and setup for Scenario B?

**Expected Keywords**: german, russian, setup, hex
**Expected Rule**: None

### Agent Answer (571.4s)
For Scenario B, the forces and setup instructions are not directly specified in the given rules. The provided text does not contain specific information on the units or their deployment for Scenario B. [No applicable rule found]

**Debug**: Retrieved=8, XRefs=4
**Top Sources**:
- pdfcoffee.com_asl-journal-9-pdf-free.pdf (P6) Rule: 
- ScenarioC.pdf (P8) Rule: 
- ScenarioB.pdf (P8) Rule: 

---

## Case 11 [scenario]
**Query**: What special balance adjustments apply to scenario A according to the Nov 2025 balance document?

**Expected Keywords**: balance, adjustment
**Expected Rule**: None

### Agent Answer (395.7s)
The provided rules text does not contain any specific balance adjustments for scenario A according to the Nov 2025 balance document. [Scenario Balance updates do not mention Scenario A]

**Debug**: Retrieved=8, XRefs=1
**Top Sources**:
- ASL_Scenario_Balance_Nov_2025.pdf (P5) Rule: 
- ASL_Scenario_Balance_Nov_2025.pdf (P5) Rule: 
- ASL_Scenario_Balance_Nov_2025.pdf (P5) Rule: 

---

## Case 12 [concept]
**Query**: How does Bypass movement work in ASL and when is it blocked?

**Expected Keywords**: bypass, hexside, building, woods, blocked
**Expected Rule**: A4.31

### Agent Answer (676.7s)
In Advanced Squad Leader (ASL), Bypass movement allows units to traverse a hexside without occupying it fully. It is blocked under specific conditions:

1. **Blocked by Certain Features:** A unit may not bypass over an impassable terrain feature [2.34].

2. **Occupied Hexsides:** If the exit or entry hexside of a hex is occupied by another vehicle, Bypass is blocked and the moving vehicle must straddle that hexside to continue its movement [2.34]. This means vehicles cannot bypass through hexsides where other vehicles are positioned.

Additionally, while not directly about blocking Bypass, vehicles in ASL may voluntarily end their MPh using Bypass but remain straddled across the last traversed hexside with its VCA and vehicular Target Facing defined [2.34].

There is no explicit rule provided here that states units cannot bypass over enemy-occupied or impassable terrain directly within a hex, but standard movement rules in ASL generally prevent such actions unless otherwise specified.

[Sources: 2.34, and implied from overall mechanics of Bypass as presented]

**Debug**: Retrieved=8, XRefs=8
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.33
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.34
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.32

---

## Case 13 [concept]
**Query**: What is a Fire Group (FG) in ASL and how is it formed?

**Expected Keywords**: fire group, fg, firepower, combine
**Expected Rule**: None

### Agent Answer (560.7s)
A Fire Group (FG) in ASL is formed when two or more units/weapons join together to make a combined fire attack [7.5]. Two SMC manning the same SW are not considered a FG as they are treated as one firing unit [7.5]. Additionally, for units from different Locations to form part of the same FG, each Location must be adjacent to another participating unit's Location within the FG [7.5]. It is possible to have a FG composed of many adjacent Locations provided that each contains a Personnel unit participating in the attack [7.5]. A leader alone in a Location cannot serve as a link in a FG unless he is Heroic or firing a SW because leaders usually lack attack capability [7.5]. Lastly, units inside a pillbox cannot form a FG with units outside the pillbox [7.5].

No errata affects this base rule.

**Debug**: Retrieved=8, XRefs=0
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 7.5
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 

---

## Case 14 [concept]
**Query**: What does CX mean in ASL and how does it affect movement?

**Expected Keywords**: cx, encumbered, movement, mf, hmg
**Expected Rule**: None

### Agent Answer (6266.8s)
The provided rules texts do not contain information about what "CX" means or how it affects movement in ASL. There is no direct reference to CX in the given sections from the core rules or other documents. Therefore, I cannot provide an answer based solely on the information you have provided. [No citation available]

**Debug**: Retrieved=8, XRefs=0
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 

---

## Case 15 [situation]
**Query**: A Russian squad with an HMG is CX and lacks 4 MF to enter a hex. Can they still enter it?

**Expected Keywords**: minimum, move, yes, mf, cx
**Expected Rule**: A4.134

### Agent Answer (591.9s)
A Russian squad with an HMG that is CX (Casualty) can still enter the hex if it lacks 4 MF by using a Minimum Movement of 1 MF (as per [20.65]) and then using a second Impulse to remove its CX status, but directly, according to the rules provided: they may use a Minimum Move of 1 MP to enter an adjacent hex even when disrupted or pinned [4.13], and can attempt to recover from CX during their next Impulse [20.65]. However, since the specific scenario involves being CX and lacking exactly 4 MF short (beyond standard disruption penalties), they would first need to utilize a Minimum Move of 1 MP to enter an adjacent hex as per [4.13] and then address the CX status in their next Impulse.

To directly answer if they can still enter with only lacking 4 MP: Yes, under the rules provided, they may use their remaining Movement Points (MP) to perform a Minimum Move of 1 MP to an adjacent hex despite being CX [4.13], though further movement would be contingent on resolving their CX status in subsequent impulses [20.65].

**Debug**: Retrieved=8, XRefs=0
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.134
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 11.52
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 11.4

---

## Case 16 [situation]
**Query**: A vehicle is hit and the Final DR equals the Kill Number. What happens to the vehicle and its PRC?

**Expected Keywords**: immobilized, hd, unaffected, prc
**Expected Rule**: A7.308

### Agent Answer (468.4s)
When a vehicle is hit and the Final DR equals the Kill Number:

1. The vehicle is destroyed [6.71].
2. Its PRC (Passenger/Rider Crew) are eliminated with no chance to roll for survival unless they were Riders in CC, who can be attacked separately as Infantry-vs-Infantry CC attack [11.611]. Other PRC must share the fate of their vehicle.
3. If the vehicle is destroyed by a direct fire (DF) or mortar/area target type that does not result in burning, surviving PRC (excluding Riders in CC) roll for Survival individually [9.2].
4. No separate check for immobility occurs since destruction eliminates the vehicle outright.

There are no errata entries provided that override these rules based on the given text.

**Debug**: Retrieved=8, XRefs=2
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 7.309
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 11.501
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 11.611

---

## Case 17 [situation]
**Query**: Two wadi overlays on different boards are adjacent to each other. Are their common hexsides treated as wadi hexsides?

**Expected Keywords**: yes, wadi, adjacent, hexside
**Expected Rule**: A2.76

### Agent Answer (374.1s)
According to Rule 2.76 [2.76], if two wadi overlays on different boards are adjacent to each other, their common hexsides are treated as wadi hexsides, except where a wadi cliff is present.

**Debug**: Retrieved=8, XRefs=1
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.76
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.74
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 2.72

---

## Case 18 [multi_hop]
**Query**: In Scenario A, the German forces set up first. What does the base infantry setup rule say, and how does this scenario modify the standard procedure?

**Expected Keywords**: german, sets up first, scenario, setup
**Expected Rule**: None

### Agent Answer (524.8s)
In Scenario A, German forces set up first according to standard procedure where setup typically follows the sequence determined by scenario-specific instructions. The base infantry setup rule states that units must be placed on the map according to their scenario-specific deployment zones and any additional rules provided [A17.2]. 

For Scenario A specifically:
- Each German infantry unit must take a TC after "At Start" placement, with failure resulting in the unit starting broken but not being subject to Desperation Morale during the initial RPh [ScenarioE.pdf | Priority: P8].

No modifications or errata from the provided rules alter this standard procedure for Scenario A.

**Debug**: Retrieved=8, XRefs=3
**Top Sources**:
- ScenarioE.pdf (P8) Rule: 
- ScenarioE.pdf (P8) Rule: 
- ScenarioF.pdf (P8) Rule: 

---

## Case 19 [multi_hop]
**Query**: A unit performs a Bypass move past a building that touches a wall/hedge depiction. Is Bypass blocked, and what rules govern this interaction?

**Expected Keywords**: bypass, blocked, wall, hedge, building, hexside
**Expected Rule**: A4.31

### Agent Answer (722.0s)
When a unit performs a Bypass move past a building that touches a wall/hedge depiction, the interaction with the wall/hedge is governed by Rule [D7.3] which states that Bypass cannot cross through Locations or hexsides blocked by obstacles like walls/hedges. However, for specific cases involving Bypass and adjacent buildings with walls/hedges, Rules [F5.21] and [F8.16] provide clarification: if a unit attempts to bypass an obstacle (wall/hedge) directly between two Locations, the move is not blocked provided the unit does not enter the Location containing the wall/hedge depiction.

To summarize:
- Bypass cannot cross through Locations or hexsides explicitly blocked by walls/hedges [D7.3].
- However, if a building touches a wall/hedge and the unit bypasses adjacent to it without entering that location, the move is possible under specific movement rules governing adjacency and obstacle interaction [F5.21], [F8.16].

**Debug**: Retrieved=8, XRefs=10
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.31
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 4.3
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 12.151

---

## Case 20 [qa_source]
**Query**: According to the Q&A document, what is the official source for each clarification, and which items are unofficial?

**Expected Keywords**: official, unofficial, source, square brackets
**Expected Rule**: None

### Agent Answer (792.5s)
For clarifications from Q&A documents:
- Official sources are cited when answers reference [Gen], [An], [Mw] or specific edition years [e.g., An93a].
- Unofficial items generally lack these references, using phrases like "See prior entry" or cross-referencing to other rules without a source document.

Specific clarifications:
1. A10.31 & A19.13: Official sources are Gen25.6, An95w, An96, Mw.
   - [A10.31 & A19.13]
   
2. Finnish Leader Replacement (A19.12): Answer cites An89, An95w, An96, Mw as official sources.
   - [A19.12]

3. Italian HS Replacement: Cited from An93b, An96 officially.
   - [A19.13 & A10.31]
   
4. Ammunition Shortage (A19.131): Officially cited from An92, An95w, An96, Mw.
   - [A19.131]

5. Splitting MMCs: Rule 15.44 from core rules is official.
   - [Rule 15.44]

Items lacking specific citations or referencing other Q&A entries (e.g., "See prior entry") are unofficial.

Sources explicitly referenced in the Q&A document indicate official status, while those without a citation should be considered unofficial interpretations based on community consensus or house rules.

**Debug**: Retrieved=8, XRefs=4
**Top Sources**:
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 8.11
- pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf (P1) Rule: 10.52
- SR ASL_QA v22.pdf (P6) Rule: 

---

