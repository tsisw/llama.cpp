// Ground-truth reference via libllama: tokenize a prompt, run llama_decode, print the prompt
// token ids and the greedy argmax of the last position's logits. Feed the printed ids to
// recon_cpu_check to compare build_layer against this.
#include "llama.h"
#include <cstdio>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    const char * model = argv[1];
    const char * prompt = argc > 2 ? argv[2] : "hello world";
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(model, mp);
    const llama_vocab * vocab = llama_model_get_vocab(m);

    std::vector<llama_token> toks(64);
    int n = llama_tokenize(vocab, prompt, (int) std::string(prompt).size(), toks.data(), (int) toks.size(), true, false);
    toks.resize(n);
    printf("ids:"); for (int t : toks) printf(" %d", t); printf("\n");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(m, cp);

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    const float * logits = llama_get_logits_ith(ctx, n - 1);
    int nv = llama_vocab_n_tokens(vocab);
    int best = 0; float bv = logits[0];
    for (int v = 1; v < nv; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
    char buf[256]; int L = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
    buf[L > 0 ? L : 0] = 0;
    printf("REFERENCE next token: %d  '%s'  logit=%.5f\n", best, buf, bv);
    return 0;
}
