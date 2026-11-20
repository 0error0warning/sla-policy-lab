"""CPU-only correction for the official assistant-message ending delimiter."""

from .wire import final_tokens


def assistant_message(ids, thinking, close_id, open_id, eos_ids, chat_end_id):
    """Stop at the first official chat end; never salvage a later message's JSON."""
    ids = list(ids)
    position = ids.index(chat_end_id) if chat_end_id in ids else None
    message = ids if position is None else ids[:position]
    final, channel, thoughts = final_tokens(
        message, thinking, close_id, open_id, eos_ids
    )
    return final, channel, thoughts, position
