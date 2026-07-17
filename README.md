## Layers

```python
RMSNorm(X:[B, L, D]):  --> [B, L, D]
  Require: 
    y: [D]
  
  Steps:
    rms = sqrt(mean(X**2 + e, dim=-1)/D)
    X^ = X / rms
    
    return x^ @ y  # y:[D] --> [B, l, D] Broadcasting across seq length and batches
````


```python
LearnedEmbedding(X:[B, L]): --> [B, L, D]
  Require: 
    E: [VOCAB_SIZE, D] # Embedding Matrix
    P: [MAX_LEN, D] # Position Matrix
  
  Steps:
    I = [0, 1, ..., L-1]
    X_pos = P[I] # X_pos: [L, D]
    X_tok = E[X] # X_emb: [B, L, D]
    
    return X_pos + X_tok
````

```python
Multi_Head_Attention(X:[B, L, D]): --> [B, L, D]
  Require: 
    W_Q: [D, D] # Query Matrix
    W_K: [D, D] # Key Matrix
    W_V: [D, D] # Key Matrix
    W_O: [D, D] # Key Matrix
  
  Steps:
    Q = W_Q @ X
    K = W_K @ X
    V = W_V @ X
    
    for n in Num_heads:
        Scores = softmax((Qn@Kn.T)/sqrt(Dn) + M) # [B, L, L]
        #                   -- Here M is causal mask :[L, L] where Mij = 0 if j<=(L-i) else -inf
        Values = Scores @ Vn # [B, L, Dn]
    Attention = Values.concat()
    
    return Attention @ W_O
````

```python
FeedForward(X:[B, L, D]): --> [B, L, D]
  Require: 
    Wup: [D, D_H] # Up Matrix
    Wgate: [D, D_H] # Gate Matrix
    Wdown: [D_H, D] # Down Matrix
  
  Steps:
    temp = SiLU(Wup @ X) * Wgate @ X
    y = Wdown @ temp

    return temp
````

## Block and Model

```python
Block(X:[B, L, D]): --> [B, L, D]

  Steps:
    X = X + Attention(LayerNorm(X)) # Pre-normallization
    X = X + FeedForward(LayerNorm(X))

    return X
````


```python
Model(X:[B, L]): --> [B, L, D]
  Require:
    W_head: [D, VOCAB_SIZE]
    
  Initiallization:
    Model.layers = normal_distribution(layers.weight, mean=0, std=0.02)
    
  Steps:
    X = LearnedEmbedding(X)
    for block in Blocks*Num_Layers:
      X = block(X)
    
    output = W_head(X)
    return output
````