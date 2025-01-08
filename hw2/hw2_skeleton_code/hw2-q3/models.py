import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # For encoder hidden states
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)  # For decoder hidden states
        self.v = nn.Linear(hidden_size, 1, bias=False)  # Scoring vector
        self.W_out = nn.Linear(hidden_size*2,  hidden_size, bias=False)  # Scoring vector
    def forward(self, query, encoder_outputs, src_lengths):
        """
        Args:
            query: (batch_size, 1, hidden_size) - decoder hidden state for the current step
            encoder_outputs: (batch_size, max_src_len, hidden_size) - all encoder hidden states
            src_lengths: (batch_size) - lengths of the source sequences
        Returns:
            context_vector: (batch_size, hidden_size) - attention-weighted sum of encoder states
            attention_weights: (batch_size, max_src_len) - normalized attention scores
        """
        query = query.unsqueeze(1)

        encoder_scores = self.W_h(encoder_outputs)  # (batch_size, max_src_len, hidden_size)
        query_scores = self.W_s(query)  # (batch_size, 1, hidden_size)

        scores = self.v(torch.tanh(encoder_scores + query_scores))  # (batch_size, max_src_len, 1)

        mask = self.sequence_mask(src_lengths)
        scores.masked_fill_(~mask.unsqueeze(2), float("-inf"))

        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, max_src_len)

        context_vector = torch.bmm(attention_weights.transpose(1,2), encoder_outputs)  # (batch_size, 1, hidden_size)
        
        w_out = torch.tanh(self.W_out(torch.cat([context_vector, query], 2)))
        return  w_out, attention_weights

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        embedded_src = self.embedding(src)
        embedded_src = self.dropout(embedded_src)

        # Pack padded sequence
        packed_src = pack(embedded_src, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_src)

        # Unpack the output
        output, _ = unpack(packed_output, batch_first=True)

        # Apply dropout to the output
        output = self.dropout(output)

        return output, (hidden, cell)

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(self, tgt, dec_state, encoder_outputs, src_lengths):
        embedded_tgt = self.embedding(tgt)
        embedded_tgt = self.dropout(embedded_tgt)

        outputs = []
        attention_weights = []

        for t in range(tgt.size(1)):  # Loop over each time step
            input_t = embedded_tgt[:, t, :].unsqueeze(1)  # (batch_size, 1, hidden_size)
            output, dec_state = self.lstm(input_t, dec_state)

            if self.attn is not None:
                print("\n\n\n\n\n\n\n\n\n\n\n\n")
                # Apply attention mechanism
                output, attn_weights = self.attn(
                    output.squeeze(1), encoder_outputs, src_lengths
                )
                attention_weights.append(attn_weights)

                # Combine context vector with LSTM output
                #output = torch.cat([output.squeeze(1), context_vector], dim=-1)

            outputs.append(output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_tgt_len, hidden_size)

        if self.attn is not None:
            attention_weights = torch.stack(attention_weights, dim=1)

        return outputs, dec_state, attention_weights

"""
class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
"""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)
        self.generator.weight = self.decoder.embedding.weight

    def forward(self, src, src_lengths, tgt, dec_hidden=None):
        # Pass through the encoder
        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        # Reshape the encoder hidden state if it's bidirectional
        if final_enc_state[0].shape[0] == 2:  # If bidirectional
            dec_hidden = reshape_state(final_enc_state)
        else:
            dec_hidden = final_enc_state

        # Pass through the decoder

        output, dec_hidden, _= self.decoder(tgt, dec_hidden, encoder_outputs, src_lengths)

        return self.generator(output), dec_hidden
