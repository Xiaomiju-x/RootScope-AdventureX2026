# FAQ

## Why not just use YOLO?

The project focuses on evidence agreement and authority separation. Semantic
classification is only one evidence branch; an independent geometric branch and
deterministic safety gate can force `HOLD`.

## Does the LLM control the pump?

No. The local LLM and RAG are read-only explanation components. Changing their
text does not change the public action proposal, which is covered by a test.

## Is the LLM running on the BPU?

No. The local language-model paths run on the RDK X5 CPU. BPU work is associated
with visual acceleration qualification and auxiliary evidence.

## Is this repository enough to build the robot?

No. It is a safe open-core reference. Model weights, data, thresholds, firmware,
hardware protocol and deployment material are not included.

## Why release code if the core stays private?

The public implementation makes the safety philosophy reviewable: evidence
conflicts fail closed, LLM output has zero authority, and the result is only an
abstract proposal. The parts that would enable direct reproduction or unsafe
physical operation remain private.

