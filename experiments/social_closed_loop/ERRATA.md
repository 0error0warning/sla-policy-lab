# Frozen-protocol wording clarification

The first paragraph of PROTOCOL.md says P86 "只接受只读测试". That wording is inaccurate. P86 never participates in the simulated environment and supplies no personal feedback, but its prompts **are included in the supervised training batches**, using targets derived from the other partners' evidence. This inclusion is specified later in the same protocol and implemented identically for LoRA and the routed readout.

P86 is therefore a registered identity with no personal observations, rather than an unseen test-only identity. Changes in its predictions measure the effects of explicit group-target supervision. They do not demonstrate spontaneous zero-shot identity transfer. The frozen protocol file and original run package are retained; this clarification does not change the actual experiment or discard data.
