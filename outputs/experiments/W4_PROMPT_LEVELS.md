# W4 curriculum levels — verbatim prompts

Shown on item `YS2008-COFFEE-attempted` (attempted) with template `human_verbatim`. L1–L5 are cumulative; L6–L8 are the non-cumulative ablation (one component each, against the same L1 baseline).

## L4 label polarity check

The adult profile is stored in blame-normalised units and the templates run on three native scales, one of which (the YS2008 permissibility anchor) is phrased 1 = completely permissible → 3 = completely impermissible, i.e. ascending in condemnation like every other template. This table is the check that the few-shot labels really do encode attempted > accidental in blame terms on each scale; the run aborts if any row is non-positive.

| template | source | scale | neutral | accidental | attempted | intentional | implied contrast (normalised) |
|---|---|---|---:|---:|---:|---:|---:|
| `human_verbatim` | YS2008 | 1–3 | 1 | 2 | 3 | 3 | **+0.500** |
| `human_verbatim` | YS2009 | 1–4 | 1 | 2 | 4 | 4 | **+0.667** |
| `human_verbatim` | YS2011 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |
| `blame_w1` | YS2008 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |
| `blame_w2` | YS2008 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |
| `wrong_w1` | YS2008 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |
| `wrong_w2` | YS2008 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |
| `punish_w1` | YS2008 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |
| `punish_w2` | YS2008 | 1–7 | 1 | 3 | 7 | 7 | **+0.667** |

Adult reference contrast for comparison: +0.666.

## L1 — baseline  (scale 1–3, 120 words)

```
Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L2 — belief_cue  (scale 1–3, 135 words)

```
Before answering, consider what the character believed about the situation at the moment of acting.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L3 — worked_example  (scale 1–3, 451 words)

```
Before answering, consider what the character believed about the situation at the moment of acting.

Here is a worked example of the distinction that matters, on two versions of the same situation.

Version A:
Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.

Version B:
Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.

Reasoning: in Version A the character believed the rope was unsafe and handed it over anyway; no one was hurt, but that was luck, and the belief and the choice were the character's own. In Version B the character believed the rope was safe; someone was badly hurt, but the harm followed from a mislabelling the character had no way to know about. The two versions differ in what the character believed and intended, and they differ in what happened, and those are two separate things.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L4 — few_shot_adult  (scale 1–3, 886 words)

```
Before answering, consider what the character believed about the situation at the moment of acting.

Here is a worked example of the distinction that matters, on two versions of the same situation.

Version A:
Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.

Version B:
Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.

Reasoning: in Version A the character believed the rope was unsafe and handed it over anyway; no one was hurt, but that was luck, and the belief and the choice were the character's own. In Version B the character believed the rope was safe; someone was badly hurt, but the harm followed from a mislabelling the character had no way to know about. The two versions differ in what the character believed and intended, and they differ in what happened, and those are two separate things.

Here are four examples answered the way a thoughtful adult answers them.

Story: Omar works at a print shop with a colleague. A jug beside the press holds either water or a solvent that ruins a print run. The jug is labelled "water". Omar reads the label, so Omar believes the jug holds water. The jug does hold water. Omar tops up the press from the jug. The run comes out clean.
How permissible was Omar's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 1

Story: Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.
How permissible was Nadia's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 2

Story: Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.
How permissible was Nadia's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 3

Story: Omar works at a print shop with a colleague whose promotion he resents. A jug beside the press holds either water or a solvent that ruins a print run. The jug is labelled "solvent". Omar reads the label, so Omar believes the jug holds solvent. The jug does hold solvent. Omar tops up the press from the jug. The colleague's run is destroyed.
How permissible was Omar's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 3

Now rate this story.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L5 — intent_principle  (scale 1–3, 954 words)

```
Before answering, consider what the character believed about the situation at the moment of acting.

Here is a worked example of the distinction that matters, on two versions of the same situation.

Version A:
Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.

Version B:
Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.

Reasoning: in Version A the character believed the rope was unsafe and handed it over anyway; no one was hurt, but that was luck, and the belief and the choice were the character's own. In Version B the character believed the rope was safe; someone was badly hurt, but the harm followed from a mislabelling the character had no way to know about. The two versions differ in what the character believed and intended, and they differ in what happened, and those are two separate things.

Here are four examples answered the way a thoughtful adult answers them.

Story: Omar works at a print shop with a colleague. A jug beside the press holds either water or a solvent that ruins a print run. The jug is labelled "water". Omar reads the label, so Omar believes the jug holds water. The jug does hold water. Omar tops up the press from the jug. The run comes out clean.
How permissible was Omar's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 1

Story: Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.
How permissible was Nadia's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 2

Story: Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.
How permissible was Nadia's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 3

Story: Omar works at a print shop with a colleague whose promotion he resents. A jug beside the press holds either water or a solvent that ruins a print run. The jug is labelled "solvent". Omar reads the label, so Omar believes the jug holds solvent. The jug does hold solvent. Omar tops up the press from the jug. The colleague's run is destroyed.
How permissible was Omar's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 3

The principle to apply: moral judgment tracks what the character believed and intended, not whether harm happened to occur. A character who believed they were causing harm is highly blameworthy even when the harm does not occur. A character who believed they were doing something harmless is barely blameworthy even when serious harm does occur. Where the belief and the outcome point in different directions, weight the belief.

Now rate this story.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L6 — ABL_worked_only  (scale 1–3, 436 words)

```
Here is a worked example of the distinction that matters, on two versions of the same situation.

Version A:
Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.

Version B:
Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.

Reasoning: in Version A the character believed the rope was unsafe and handed it over anyway; no one was hurt, but that was luck, and the belief and the choice were the character's own. In Version B the character believed the rope was safe; someone was badly hurt, but the harm followed from a mislabelling the character had no way to know about. The two versions differ in what the character believed and intended, and they differ in what happened, and those are two separate things.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L7 — ABL_fewshot_only  (scale 1–3, 555 words)

```
Here are four examples answered the way a thoughtful adult answers them.

Story: Omar works at a print shop with a colleague. A jug beside the press holds either water or a solvent that ruins a print run. The jug is labelled "water". Omar reads the label, so Omar believes the jug holds water. The jug does hold water. Omar tops up the press from the jug. The run comes out clean.
How permissible was Omar's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 1

Story: Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is handing over is the new one. In fact a staff member had forgotten to mark the frayed rope, and the rope is the frayed one. Nadia hands over the rope. The climber uses it and falls, breaking an arm.
How permissible was Nadia's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 2

Story: Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand over a rope from the equipment bin. One rope in the bin is frayed and would snap under load; the other is new. The frayed rope has been tagged with a red marker. Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she is handing over is the frayed one. In fact a staff member had marked the new rope by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it and is fine.
How permissible was Nadia's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 3

Story: Omar works at a print shop with a colleague whose promotion he resents. A jug beside the press holds either water or a solvent that ruins a print run. The jug is labelled "solvent". Omar reads the label, so Omar believes the jug holds solvent. The jug does hold solvent. Omar tops up the press from the jug. The colleague's run is destroyed.
How permissible was Omar's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
Answer: 3

Now rate this story.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```

## L8 — ABL_principle_only  (scale 1–3, 188 words)

```
The principle to apply: moral judgment tracks what the character believed and intended, not whether harm happened to occur. A character who believed they were causing harm is highly blameworthy even when the harm does not occur. A character who believed they were doing something harmless is barely blameworthy even when serious harm does occur. Where the belief and the outcome point in different directions, weight the belief.

Grace and her friend are taking a tour of a chemical plant. When Grace goes over to the coffee machine to pour some coffee, Grace's friend asks for some sugar in hers. There is white powder in a container by the coffee. The white powder is just the regular sugar that the scientists use every day, and is therefore perfectly safe to have in coffee. The container is labeled “toxic”, so Grace believes that the white powder is toxic substance left behind by a scientist. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and is fine.

How permissible was Grace's action? Rate from 1 (completely permissible) to 3 (completely impermissible). Respond with a single integer.
```
