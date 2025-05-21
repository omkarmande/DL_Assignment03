import os
import numpy as np
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

class TransliterationSystem:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.input_tokenizer = None
        self.target_tokenizer = None
        
    def _load_dataset(self, filename):
        """Load TSV file into list of pairs"""
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as f:
            return [line.strip().split('\t') for line in f if '\t' in line]
    
    def prepare_data(self):
        """Load and preprocess training/validation data"""
        train_data = self._load_dataset("mr.translit.sampled.train.tsv")
        val_data = self._load_dataset("mr.translit.sampled.dev.tsv")
        
        # Process text pairs
        train_source = [x[1] for x in train_data]
        train_target = [x[0] for x in train_data]
        
        # Create tokenizers
        self.input_tokenizer = self._create_tokenizer(train_source + [x[1] for x in val_data])
        self.target_tokenizer = self._create_tokenizer(
            ['\t' + t for t in train_target] + 
            [t + '\n' for t in train_target] +
            ['\t' + t for t in [x[0] for x in val_data]] + 
            [t + '\n' for t in [x[0] for x in val_data]]
        )
        
        # Prepare sequences
        max_source_len = max(len(s) for s in train_source)
        max_target_len = max(len(t) for t in train_target) + 1  # +1 for start/end tokens
        
        train_enc = self._texts_to_padded_sequences(train_source, self.input_tokenizer, max_source_len)
        train_dec_in = self._texts_to_padded_sequences(['\t' + t for t in train_target], self.target_tokenizer, max_target_len)
        train_dec_out = np.expand_dims(
            self._texts_to_padded_sequences([t + '\n' for t in train_target], self.target_tokenizer, max_target_len),
            -1
        )
        
        return (train_enc, train_dec_in, train_dec_out), self.input_tokenizer, self.target_tokenizer
    
    def _create_tokenizer(self, texts):
        """Create character-level tokenizer"""
        tokenizer = preprocessing.text.Tokenizer(char_level=True, lower=False)
        tokenizer.fit_on_texts(texts)
        return tokenizer
    
    def _texts_to_padded_sequences(self, texts, tokenizer, max_len):
        """Convert texts to padded sequences"""
        seq = tokenizer.texts_to_sequences(texts)
        return preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=max_len)

class Seq2SeqModelBuilder:
    """Builds sequence-to-sequence models with different RNN types"""
    
    RNN_TYPES = {
        'rnn': layers.SimpleRNN,
        'lstm': layers.LSTM,
        'gru': layers.GRU
    }
    
    def __init__(self, input_vocab_size, target_vocab_size):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
    
    def build_model(self, rnn_type='lstm', embedding_dim=256, hidden_dim=512, 
                   enc_layers=1, dec_layers=1, dropout=0.2):
        """Build end-to-end seq2seq model"""
        
        # Input layers
        encoder_input = layers.Input(shape=(None,))
        decoder_input = layers.Input(shape=(None,))
        
        # Shared embedding
        embedding = layers.Embedding(self.input_vocab_size, embedding_dim, mask_zero=True)
        
        # Encoder
        enc_output, enc_states = self._build_encoder(
            encoder_input, embedding, rnn_type, hidden_dim, enc_layers, dropout)
        
        # Decoder
        decoder_output = self._build_decoder(
            decoder_input, embedding, rnn_type, hidden_dim, dec_layers, dropout, enc_states)
        
        # Final output
        output = layers.Dense(self.target_vocab_size, activation='softmax')(decoder_output)
        
        return models.Model([encoder_input, decoder_input], output)
    
    def _build_encoder(self, inputs, embedding, rnn_type, hidden_dim, num_layers, dropout):
        """Build encoder architecture"""
        x = embedding(inputs)
        rnn_class = self.RNN_TYPES[rnn_type.lower()]
        
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            x = rnn_class(
                hidden_dim,
                return_sequences=return_sequences,
                return_state=True,
                dropout=dropout,
                name=f'enc_{rnn_type}_{i}'
            )(x)
        
        return x if num_layers == 1 else x[0], x[1:] if num_layers > 1 else x[1]
    
    def _build_decoder(self, inputs, embedding, rnn_type, hidden_dim, num_layers, dropout, initial_state):
        """Build decoder architecture"""
        x = embedding(inputs)
        rnn_class = self.RNN_TYPES[rnn_type.lower()]
        
        for i in range(num_layers):
            x = rnn_class(
                hidden_dim,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                name=f'dec_{rnn_type}_{i}'
            )(x, initial_state=initial_state)
        
        return x[0]

class InferenceSystem:
    """Handles model inference with beam search"""
    
    def __init__(self, model, rnn_type, hidden_dim):
        self.rnn_type = rnn_type.lower()
        self._setup_inference_models(model, hidden_dim)
        
    def _setup_inference_models(self, model, hidden_dim):
        """Create encoder/decoder models for inference"""
        encoder_input = model.input[0]
        decoder_input = model.input[1]
        embedding = model.get_layer('embedding')
        
        # Encoder model
        enc_output = embedding(encoder_input)
        rnn_layer = next(l for l in model.layers if l.name.startswith(f'enc_{self.rnn_type}'))
        
        if self.rnn_type == 'lstm':
            _, state_h, state_c = rnn_layer(enc_output)
            self.encoder_model = models.Model(encoder_input, [state_h, state_c])
            self.state_size = 2
        elif self.rnn_type == 'gru':
            _, state_h = rnn_layer(enc_output)
            self.encoder_model = models.Model(encoder_input, [state_h])
            self.state_size = 1
        else:  # Simple RNN
            _, state_h = rnn_layer(enc_output)
            self.encoder_model = models.Model(encoder_input, [state_h])
            self.state_size = 1
        
        # Decoder model
        decoder_states_input = [
            layers.Input(shape=(hidden_dim,)) for _ in range(self.state_size)
        ]
        decoder_emb = embedding(decoder_input)
        
        decoder_rnn = next(l for l in model.layers if l.name.startswith(f'dec_{self.rnn_type}'))
        decoder_output = decoder_rnn(decoder_emb, initial_state=decoder_states_input)
        
        dense_layer = model.get_layer('dense')
        decoder_output = dense_layer(decoder_output[0])
        
        self.decoder_model = models.Model(
            [decoder_input] + decoder_states_input,
            [decoder_output] + list(decoder_output[1:])
    
    def beam_search_decode(self, input_seq, tokenizer, beam_width=3, max_len=30):
        """Decode sequence using beam search"""
        idx_to_char = {i: c for c, i in tokenizer.word_index.items()}
        idx_to_char[0] = ''
        
        start_token = tokenizer.word_index['\t']
        end_token = tokenizer.word_index['\n']
        
        states = self.encoder_model.predict(input_seq)
        if self.state_size == 1:
            states = [states]
        
        beams = [([start_token], 0.0, states)]
        
        for _ in range(max_len):
            candidates = []
            for seq, score, states in beams:
                if seq[-1] == end_token:
                    candidates.append((seq, score, states))
                    continue
                
                target_seq = np.array([[seq[-1]]])
                outputs = self.decoder_model.predict([target_seq] + states)
                probs = outputs[0][0, -1, :]
                top_tokens = np.argsort(probs)[-beam_width:]
                
                for token in top_tokens:
                    new_score = score - np.log(probs[token] + 1e-9)
                    candidate_seq = seq + [token]
                    candidates.append((candidate_seq, new_score, outputs[1:]))
            
            beams = sorted(candidates, key=lambda x: x[1])[:beam_width]
        
        best_seq = beams[0][0]
        return ''.join(idx_to_char.get(i, '') for i in best_seq[1:-1])

def run_sweep():
    """Configure and run hyperparameter sweep"""
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'rnn_type': {'values': ['rnn', 'lstm', 'gru']},
            'embedding_dim': {'values': [64, 128, 256]},
            'hidden_dim': {'values': [128, 256, 512]},
            'enc_layers': {'values': [1, 2]},
            'dec_layers': {'values': [1, 2]},
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'batch_size': {'values': [32, 64]},
            'beam_width': {'values': [1, 3, 5]}
        }
    }
    
    def sweep_train():
        with wandb.init() as run:
            config = run.config
            run.name = (f"{config.rnn_type}_e{config.embedding_dim}_h{config.hidden_dim}_"
                       f"enc{config.enc_layers}_dec{config.dec_layers}_"
                       f"drop{config.dropout}_beam{config.beam_width}")
            
            # Initialize system
            system = TransliterationSystem("/kaggle/input/dakshinadataset/dakshina_dataset_v1.0/mr/lexicons")
            (train_enc, train_dec_in, train_dec_out), _, target_tokenizer = system.prepare_data()
            
            # Build model
            builder = Seq2SeqModelBuilder(
                len(system.input_tokenizer.word_index) + 1,
                len(target_tokenizer.word_index) + 1
            )
            model = builder.build_model(
                rnn_type=config.rnn_type,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                enc_layers=config.enc_layers,
                dec_layers=config.dec_layers,
                dropout=config.dropout
            )
            
            model.compile(
                optimizer=Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(
                [train_enc, train_dec_in],
                train_dec_out,
                batch_size=config.batch_size,
                epochs=10,
                validation_split=0.1,
                callbacks=[WandbMetricsLogger(), WandbModelCheckpoint("models")],
                verbose=2
            )
    
    sweep_id = wandb.sweep(sweep_config, project="marathi-transliteration")
    wandb.agent(sweep_id, function=sweep_train, count=15)

if __name__ == "__main__":
    wandb.login(key="your_key")
    run_sweep()
