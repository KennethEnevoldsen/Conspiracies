
def forward_pass(ds,
                 model,
                 tokenizer,
                 device,
                 batch_size,
                 model_parallel,
                 ):
    """
    does the forward pass on the dataset
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    if model_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    

    ds = ds.map(__tokenizer, batched=True)
    ds.set_format(format="pt", columns=['attention_mask', 'input_ids'])

    def collate_fn(examples):
        # on the fly padding
        return tokenizer.pad(examples, return_tensors='pt')
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=collate_fn, batch_size=batch_size)

    # do forward pass
    forward_batches = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch.to(device)
            outputs = model(**batch, output_attentions=True)
            forward_batch = {
                "attention_weights": [a.to("cpu") for a in outputs.attentions],
                "embedding": outputs[0].to("cpu")}
            forward_batches.append(forward_batch)
    ds.reset_format()
    return ds, forward_batches


d_config = {
    "transformer": "Maltehb/danish-bert-botxo",
    "model_parallel": True,
    "batch_size": 1024,
    "device": torch.device("cuda")
}

# FORSØG MED AT TILFØJE EN TRANSFORMER TIL PIPELINEN


@Language.factory("custom_forward_pass", default_config=d_config)
def custom_forward_pass(doc,
                        transformer: str,
                        model_parallel: bool = True,
                        batch_size: int = 1024,
                        device: Optional[torch.device] = None,
                        ):
    # do stuff
    return doc


def add_forward(nlp: spacy.language.Language):
    return nlp.add_pipe("custom_forward_pass", name="custom_forward_pass",
                        last=True)


class ADoc(spacy.tokens.Doc):
    """
    A subclass of the SpaCy doc an forward pass of a transformer
    attached to it
    """

    def __init__(self,
                 texts: list,
                 nlp: spacy.language.Language,
                 transformer: str,
                 model_parallel=True,
                 batch_size: int = 1024,
                 device: Optional[torch.device] = None,
                 ):
        """
        transformer: a string refering to a huggingface transformer (and
        tokenizer)

        Example:
        >>> transformer = "Maltehb/danish-bert-botxo"
        """
        self = nlp(texts)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)
        # self.model = transformers.ElectraModel.from_pretrained(transformer)
        # ds = datasets.Dataset.from_dict({"text": texts})
        # sent_ds, forward_batches = forward_pass(ds, model)
        # attentions = unwrap_attention_from_batch(forward_batches)

