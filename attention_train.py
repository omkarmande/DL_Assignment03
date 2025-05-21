import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.integration.keras import WandbMetricsLogger

class AttentionLayer(layers.Layer):
    """Bahdanau Attention Layer with weight export capability"""
    def __init__(self, units, return_attention=False):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
        self.return_attention = return_attention
        self.units = units

    def call(self, query, values):
        # Add time axis for broadcasting
        query_with_time = tf.expand_dims(query, 2)  # (batch, dec_len, 1, units)
        values_with_time = tf.expand_dims(values, 1)  # (batch, 1, enc_len, units)
        
        # Attention scores calculation
        score = self.V(tf.nn.tanh(
            self.W1(values_with_time) + self.W2(query_with_time)
        ))
        
        attention_weights = tf.nn.softmax(score, axis=2)
        context_vector = tf.reduce_sum(attention_weights * values_with_time, axis=2)
        
        if self.return_attention:
            return context_vector, tf.squeeze(attention_weights, -1)
        return context_vector

class AttentionTransliterator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.input_tokenizer = None
        self.target_tokenizer = None
    
    def load_dataset(self, filename):
        """Load and parse dataset file"""
        with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
            return [line.strip().split('\t') for line in f if '\t' in line]
    
    def preprocess_data(self):
        """Load and prepare training/validation/test data"""
        train_pairs = self.load_dataset("mr.translit.sampled.train.tsv")
        val_pairs = self.load_dataset("mr.translit.sampled.dev.tsv")
        test_pairs = self.load_dataset("mr.translit.sampled.test.tsv")
        
        # Extract texts
        train_source = [x[1] for x in train_pairs]
        train_target = [x[0] for x in train_pairs]
        
        # Create tokenizers
        self.input_tokenizer = self._create_tokenizer(train_source + [x[1] for x in val_pairs])
        self.target_tokenizer = self._create_tokenizer(
            ['\t' + t for t in train_target] + 
            [t + '\n' for t in train_target]
        )
        
        # Prepare sequences
        max_source_len = max(len(s) for s in train_source)
        max_target_len = max(len(t) for t in train_target) + 1  # +1 for start/end tokens
        
        # Training data
        train_enc = self._text_to_sequence(train_source, self.input_tokenizer, max_source_len)
        train_dec_in = self._text_to_sequence(['\t' + t for t in train_target], self.target_tokenizer, max_target_len)
        train_dec_out = np.expand_dims(
            self._text_to_sequence([t + '\n' for t in train_target], self.target_tokenizer, max_target_len),
            -1
        )
        
        # Validation data
        val_enc = self._text_to_sequence([x[1] for x in val_pairs], self.input_tokenizer, max_source_len)
        val_dec_in = self._text_to_sequence(['\t' + x[0] for x in val_pairs], self.target_tokenizer, max_target_len)
        val_dec_out = np.expand_dims(
            self._text_to_sequence([x[0] + '\n' for x in val_pairs], self.target_tokenizer, max_target_len),
            -1
        )
        
        # Test data
        test_enc = self._text_to_sequence([x[1] for x in test_pairs], self.input_tokenizer, max_source_len)
        test_dec_in = self._text_to_sequence(['\t' + x[0] for x in test_pairs], self.target_tokenizer, max_target_len)
        test_dec_out = np.expand_dims(
            self._text_to_sequence([x[0] + '\n' for x in test_pairs], self.target_tokenizer, max_target_len),
            -1
        )
        
        return (train_enc, train_dec_in, train_dec_out), \
               (val_enc, val_dec_in, val_dec_out), \
               (test_enc, test_dec_in, test_dec_out)
    
    def _create_tokenizer(self, texts):
        """Create character-level tokenizer"""
        tokenizer = preprocessing.text.Tokenizer(char_level=True, lower=False)
        tokenizer.fit_on_texts(texts)
        return tokenizer
    
    def _text_to_sequence(self, texts, tokenizer, max_len):
        """Convert texts to padded sequences"""
        seq = tokenizer.texts_to_sequences(texts)
        return preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=max_len)

class AttentionModel:
    def __init__(self, input_vocab_size, target_vocab_size):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
    
    def build_model(self, embedding_dim=256, hidden_dim=512, dropout_rate=0.2):
        """Build attention-based seq2seq model"""
        # Encoder
        encoder_inputs = layers.Input(shape=(None,))
        enc_emb = layers.Embedding(self.input_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        encoder_outputs, state_h, state_c = layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True, dropout=dropout_rate
        )(enc_emb)
        
        # Decoder
        decoder_inputs = layers.Input(shape=(None,))
        dec_emb = layers.Embedding(self.target_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
        decoder_outputs = layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True, dropout=dropout_rate
        )(dec_emb, initial_state=[state_h, state_c])[0]
        
        # Attention
        context_vector, attention_weights = AttentionLayer(hidden_dim, return_attention=True)(
            decoder_outputs, encoder_outputs
        )
        concat_output = layers.Concatenate()([decoder_outputs, context_vector])
        
        # Output
        outputs = layers.Dense(self.target_vocab_size, activation='softmax')(concat_output)
        
        return models.Model([encoder_inputs, decoder_inputs], outputs)

def run_sweep():
    """Configure and execute hyperparameter sweep"""
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'embedding_dim': {'values': [128, 256]},
            'hidden_dim': {'values': [128, 256, 512]},
            'dropout_rate': {'values': [0.1, 0.2, 0.3]},
            'batch_size': {'values': [32, 64]},
            'learning_rate': {'min': 1e-4, 'max': 1e-3}
        }
    }
    
    def train():
        wandb.init()
        config = wandb.config
        
        # Initialize components
        transliterator = AttentionTransliterator(
            "/kaggle/input/dakshinadataset/dakshina_dataset_v1.0/mr/lexicons"
        )
        (train_enc, train_dec_in, train_dec_out), \
        (val_enc, val_dec_in, val_dec_out), _ = transliterator.preprocess_data()
        
        # Build model
        model = AttentionModel(
            len(transliterator.input_tokenizer.word_index) + 1,
            len(transliterator.target_tokenizer.word_index) + 1
        ).build_model(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate
        )
        
        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(
            [train_enc, train_dec_in],
            train_dec_out,
            validation_data=([val_enc, val_dec_in], val_dec_out),
            batch_size=config.batch_size,
            epochs=10,
            callbacks=[WandbMetricsLogger()],
            verbose=2
        )
        
        # Save model if performance is good
        val_acc = model.history.history['val_accuracy'][-1]
        if val_acc > 0.85:  # Adjust threshold as needed
            model.save(f"attention_model_{wandb.run.id}.keras")
    
    sweep_id = wandb.sweep(sweep_config, project="marathi-transliteration-attention")
    wandb.agent(sweep_id, function=train, count=15)

def evaluate_best_model():
    """Evaluate the best saved model on test set"""
    wandb.init(project="marathi-transliteration-attention", name="best_model_evaluation")
    
    # Initialize components
    transliterator = AttentionTransliterator(
        "/kaggle/input/dakshinadataset/dakshina_dataset_v1.0/mr/lexicons"
    )
    _, _, (test_enc, test_dec_in, test_dec_out) = transliterator.preprocess_data()
    
    # Load best model (replace with actual best run ID)
    best_model = models.load_model(
        "attention_model_BEST_RUN_ID.keras",
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    
    # Evaluate
    test_loss, test_acc = best_model.evaluate(
        [test_enc, test_dec_in],
        test_dec_out,
        verbose=1
    )
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Log results
    wandb.log({
        'test_accuracy': test_acc,
        'test_loss': test_loss
    })
    
    # Generate predictions
    preds = best_model.predict([test_enc, test_dec_in])
    pred_indices = np.argmax(preds, axis=-1)
    
    # Save predictions
    os.makedirs("predictions", exist_ok=True)
    with open("predictions/attention_predictions.txt", "w", encoding="utf-8") as f:
        for i in range(len(test_enc)):
            input_text = transliterator.input_tokenizer.sequences_to_texts([test_enc[i]])[0]
            pred_text = transliterator.target_tokenizer.sequences_to_texts([pred_indices[i]])[0]
            true_text = transliterator.target_tokenizer.sequences_to_texts([test_dec_out[i]])[0]
            f.write(f"{input_text}\t{pred_text}\t{true_text}\n")
    
    wandb.finish()

if __name__ == "__main__":
    wandb.login(key="your_api_key_here")  # Replace with your WandB key
    
    # Run either the sweep or evaluation
    run_sweep()  # Comment this out after sweep is done
    # evaluate_best_model()  # Uncomment after selecting best model
