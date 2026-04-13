# Extension Ideas for Self-Distillation Activation Oracles

## 1. Iterative Self-Distillation

**Concept**: After the initial training round, the oracle can generate descriptions 
from activations. These descriptions might capture things the original text-based 
descriptions missed (because the activations encode model-specific information).

**Pipeline**:
1. Round 1: Train oracle on (activation → text-description) pairs
2. Round 2: Use oracle to generate descriptions from activations, 
   then compare with original descriptions
3. Round 3: Train on the union of original + oracle-generated descriptions

**Why interesting**: The oracle's descriptions might capture what the model 
"knows" at each layer better than the text-prompted descriptions, because 
the oracle has learned to read the activation space directly.

## 2. Layer Comparison / Trajectory Descriptions

**Concept**: Instead of describing activations from a single layer, describe 
the *change* between layers. What information is added/removed as we go 
deeper?

**Prompt**: "Given activations from layer 9 and layer 27 of the same text, 
describe what information has been added, refined, or transformed between 
these layers."

**Training format**: Inject two sets of activations (one from each layer) 
and train the oracle to describe the differences.

## 3. Activation Arithmetic

**Concept**: Can the oracle describe the result of activation arithmetic?
For example, if we average activations from two texts, can the oracle 
describe the "blend"?

**Test**: `act("The cat sat on the mat") + act("The dog ran in the park") → ?`

## 4. Cross-Model Transfer

**Concept**: Train the oracle on Qwen3-8B activations, then test it on 
activations from a fine-tuned variant. Does the oracle still work? Can it 
describe what the fine-tuning changed?

## 5. Activation Search / Retrieval

**Concept**: Use the oracle to create a searchable index of activations. 
Given a text query, find the activations whose oracle-described content 
best matches the query.

## 6. Compositional Descriptions

**Concept**: Instead of one flat description, generate structured descriptions:
- Language: French
- Topic: Cooking
- Entities: chef, restaurant, menu
- Sentiment: positive
- Style: instructional
- Continuation: recipe instructions

This structured format might be more useful for downstream tasks and 
easier to evaluate quantitatively.

## 7. Multi-Position Relationship Descriptions

**Concept**: When injecting activations from multiple positions in the same 
text, can the oracle describe the *relationship* between them? For example:
"The activation at position 3 represents the subject, and the activation 
at position 7 represents the verb. Together they form a subject-verb 
agreement relationship."

## 8. Attention Pattern Descriptions

**Concept**: In addition to residual stream activations, inject attention 
pattern information. The oracle could then describe not just *what* the 
model represents, but *how* it's attending to different parts of the input.

## Priority Order

For this experiment session:
1. Basic single-layer oracle (current experiment)
2. Layer comparison (easy extension, informative)
3. Compositional descriptions (better evaluation)
4. Iterative self-distillation (if time permits)
