import torch
from.modules import LlamaInterpretorConfig, LlamaInterpretor
from ..utils import InterpretorModelOutput
from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
import time




class RavelInterpretorHypernetwork(nn.Module):
    # Separating the editor config file, from its base model's configurations
    def __init__(
        self,
        model_name_or_path="/work/frink/models/llama3-8B-HF",
        num_editing_heads=32,
        chop_editor_at_layer=8,
        intervention_layer=0,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()

        self.interpretor_config = LlamaInterpretorConfig.from_pretrained(model_name_or_path)
        self.interpretor_config.name_or_path = model_name_or_path
        self.interpretor_config.torch_dtype = torch_dtype
        self.interpretor_config.num_editing_heads = num_editing_heads
        self.interpretor_config.chop_editor_at_layer = chop_editor_at_layer
        self.interpretor_config.intervention_layer = intervention_layer
        self.interpretor_config._attn_implementation = 'eager'
                
        self.interpretor = LlamaInterpretor(self.interpretor_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.residual_cache = None
        self.opt = None
        self.training_loss = None
        
    def save_model(self, save_path):
        torch.save(self.interpretor.hypernetwork.state_dict(), save_path)
        
    def load_model(self, load_path):
        self.interpretor.hypernetwork.load_state_dict(torch.load(load_path))
        
    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_intervention_weight: bool = True,
        intervention_weight: torch.Tensor = None,
        inference_mode = None,
    ):
        _pred: InterpretorModelOutput = self.interpretor(
            editor_input_ids=editor_input_ids,
            editor_attention_mask=editor_input_ids != self.interpretor_config.eos_token_id,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            output_intervention_weight=output_intervention_weight,
            intervention_weight=intervention_weight,
            inference_mode=inference_mode
            # output_target_hidden_states=True,
        )
                
        if labels is not None:
            log_prob_predictions = torch.nn.functional.log_softmax(
                _pred.logits.reshape(-1, _pred.logits.shape[-1]),
                dim=1,
            )
            
            labels = labels.reshape(-1)
            assert labels.shape == log_prob_predictions.shape[:-1]
            
            # Only consider the tokens that are not -100 in target_labels
            label_indices = labels != -100
            output_idices = torch.zeros_like(label_indices)
            output_idices[:-1] = label_indices[1:]
            
            log_prob_predictions = log_prob_predictions[output_idices, :]
            labels = labels[label_indices]
            
            # Compute the cross-entropy loss with masking
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(log_prob_predictions, labels.long())
            _pred["loss"] = loss
            
        return _pred
        
    
    # Generate text using the target model, with a new edit application at every step.
    # This is a very slow way to generate text.
    # If you only want to edit first k tokens, use the forward pass instead with stop_editing_index = k
    def inspect_batch_prediction_ouptuts(self, batch, inference_mode=None):
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth"]
        self.interpretor.eval()
        
        with torch.no_grad():
            
            predictions = self.forward(
                editor_input_ids=batch["editor_input_ids"].to("cuda"),
                base_input_ids=batch["base_input_ids"].to("cuda"),
                base_attention_mask=batch["base_attention_mask"].to("cuda"),
                source_input_ids=batch["source_input_ids"].to("cuda"),
                source_attention_mask=batch["source_attention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
                output_intervention_weight=True,
                inference_mode=inference_mode
            )
            
            batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
            batch_full_output = self.tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
            
            batch_output = []
            correct = 0
            
            for label, pred_ids in zip(batch["labels"].to("cuda"), batch_pred_ids):
                
                label_idx = label != -100
                output_idx = torch.zeros_like(label_idx)
                output_idx[:-1] = label_idx[1:]
                
                label = label[label_idx]
                pred_ids = pred_ids[output_idx]
                                
                batch_output.append(
                    self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                )             
                
                correct += torch.sum(label == pred_ids) == torch.numel(label)
            
        return {
            "batch_output": batch_output,
            "batch_full_output": batch_full_output,
            "batch_intervention_weight": predictions.intervention_weight,
            "n_correct": correct,
        }
    
    def plot_heatmap(self, data_loader, idxs, batch_size=4, inference_mode=None):
        batch_id = idxs // batch_size
        example_id = idxs % batch_size

        for i, batch in enumerate(data_loader):
            if i == batch_id:
                break
            
        results = self.inspect_batch_prediction_ouptuts(batch, inference_mode=inference_mode)

        editor_input_ids = batch["editor_input_ids"][example_id]
        base_input_ids = batch["base_input_ids"][example_id]
        source_input_ids = batch["source_input_ids"][example_id]
        intervention_weight = results["batch_intervention_weight"][example_id]
        label = batch["labels"][example_id]

        assert intervention_weight.size() == (len(source_input_ids) + 1, len(base_input_ids))

        source_axis = [self.tokenizer.decode([i]) for i in source_input_ids] + ["[SELF]"]
        base_axis = [self.tokenizer.decode([i]) for i in base_input_ids]
        editor_text = self.tokenizer.decode(editor_input_ids)
        label = label[label != -100]
        label = self.tokenizer.decode(label)

        _, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(intervention_weight.float().cpu().numpy(), xticklabels=base_axis, yticklabels=source_axis, ax=ax, annot=True)

        ax.set_title(f"Instruction: {editor_text}     Label: {label}")
        ax.set_xlabel("Base Sentence Tokens")
        ax.set_ylabel("Source Sentence Tokens")
        
    
    def eval_accuracy(self, test_loader, disentangling=False, inference_mode=None):
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth"]
        
        self.interpretor.eval()
        test_loss = []
        correct_idxs = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                
                if inference_mode == "groundtruth":
                    intervention_weight = torch.zeros(len(batch["editor_input_ids"]), batch["source_input_ids"].shape[1] + 1, batch["base_input_ids"].shape[1]).to("cuda")
                    intervention_weight[:, -1, :] = 1.0
                    
                    for i in range(len(batch["base_entity_position_ids"])):
                        intervention_weight[i, -1, batch["base_entity_position_ids"][i]] = 0.0
                        intervention_weight[i, batch["source_entity_position_ids"][i], batch["base_entity_position_ids"][i]] = 1.0
                else:
                    intervention_weight=None
                    
                predictions = self.forward(
                    editor_input_ids=batch["editor_input_ids"].to("cuda"),
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    labels=batch["labels"].to("cuda"),
                    inference_mode=inference_mode,
                    intervention_weight=intervention_weight
                )
                test_loss.append(predictions["loss"].item())
                
                batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
                
                if disentangling:
                    disentangling_prediction = self.forward(
                        editor_input_ids=batch["disentangled_editor_input_ids"].to("cuda"),
                        base_input_ids=batch["disentangled_base_input_ids"].to("cuda"),
                        base_attention_mask=batch["disentangled_base_attention_mask"].to("cuda"),
                        source_input_ids=batch["disentangled_source_input_ids".to("cuda")],
                        source_attention_mask=batch["disentangled_source_attention_mask"].to("cuda"),
                        labels=batch["disentangled_labels"].to("cuda"),
                    )
                    batch_disentangling_pred_ids = torch.argmax(disentangling_prediction["logits"], dim=-1)
                    
                    correct_disentangling = []
                    
                    for label, pred_ids in zip(batch["disentangled_labels"].to("cuda"), batch_disentangling_pred_ids):
                        label_idx = label != -100
                        output_idx = torch.zeros_like(label_idx)
                        output_idx[:-1] = label_idx[1:]
                        
                        label = label[label_idx]
                        pred_ids = pred_ids[output_idx]
                        
                        correct_disentangling.append((torch.sum(label == pred_ids) == torch.numel(label)))
                    
                    correct_disentangling = torch.stack(correct_disentangling)
                    
                    test_loss[-1] += disentangling_prediction["loss"].item()
                
                for i, (label, pred_ids) in enumerate(zip(batch["labels"].to("cuda"), batch_pred_ids)):
                    label_idx = label != -100
                    output_idx = torch.zeros_like(label_idx)
                    output_idx[:-1] = label_idx[1:]
                    
                    label = label[label_idx]
                    pred_ids = pred_ids[output_idx]
                    
                    is_correct = (torch.sum(label == pred_ids) == torch.numel(label)).item()
                    if disentangling:
                        instance_idx = torch.nonzero(batch["disentangled_example_idxs"].to("cuda") == i).squeeze()
                        is_correct = is_correct and all(correct_disentangling[instance_idx])
                        
                    correct += is_correct
                    if is_correct:
                        correct_idxs.append(batch_id * len(batch["labels"]) + i)
                    total += 1
                    
        return correct / total, sum(test_loss) / len(test_loss), correct_idxs
             

    def run_train(
        self,
        train_loader,
        test_loader=None,
        epochs=1,
        eval_per_steps: int = None,
        checkpoint_per_steps: int = None,
        disentangling = False,
        lr=3e-4,
        weight_decay=0.01,
        save_dir=None,
        use_auxilary_weight_loss=False
    ):
        
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        trainable_parameters = []
        for name, param in self.named_parameters():
            if "target_model" not in name:
                trainable_parameters.append(param)
                
        self.opt = optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)  # usually: lr = 5e-5. 1e-3 worked well!
        
        total_steps = len(train_loader) * epochs
        cur_steps = 0

        for epoch in range(epochs):
            # Create a tqdm progress bar
            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                disable=True,
            ) as pbar:
                num_datapoints_in_epoch = 0
                epoch_train_loss = 0
                epoch_gradient_norm = 0
                # Train loop
                for step, batch in enumerate(
                    train_loader
                ):  
                    if eval_per_steps is not None:
                        if cur_steps % eval_per_steps == 0:
                            # Evaluate the model
                            accuracy, test_loss, _ = self.eval_accuracy(
                                test_loader, disentangling=disentangling, inference_mode=None
                            )
                            
                            if wandb.run:
                                wandb.log(
                                    {
                                        "test_average_loss": test_loss,
                                        "test_accuracy": accuracy,
                                    }
                                )
                            print(f"Accuracy: {accuracy}, Test Loss: {test_loss}")

                    if checkpoint_per_steps is not None:
                        if cur_steps % checkpoint_per_steps == 0 and save_dir is not None:
                            print("Saving model to {}".format(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}.pt")))
                            self.save_model(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}.pt"))
                            
                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size
                    self.opt.zero_grad()

                    self.prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"].to("cuda"),
                        base_input_ids=batch["base_input_ids"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask"].to("cuda"),
                        source_input_ids=batch["source_input_ids"].to("cuda"),
                        source_attention_mask=batch["source_attention_mask"].to("cuda"),
                        labels=batch["labels"].to("cuda"),
                        output_intervention_weight=True,
                        inference_mode=None
                    )

                    self.prediction_loss = self.prediction["loss"]
                    
                    if disentangling:
                        self.prediction_loss += self.forward(
                            editor_input_ids=batch["disentangled_editor_input_ids"].to("cuda"),
                            base_input_ids=batch["disentangled_base_input_ids"].to("cuda"),
                            base_attention_mask=batch["disentangled_base_attention_mask"].to("cuda"),
                            source_input_ids=batch["disentangled_source_input_ids"].to("cuda"),
                            source_attention_mask=batch["disentangled_source_attention_mask"].to("cuda"),
                            labels=batch["disentangled_labels"].to("cuda"),
                            output_intervention_weight=True,
                            inference_mode=None
                        )["loss"]
                            
                    if use_auxilary_weight_loss:
                        if cur_steps < 100:
                            intervention_weight = self.prediction.intervention_weight
                            
                            gt_weight = torch.zeros_like(intervention_weight)
                            gt_weight[:, -1, :] = 1.0
                            
                            for i in range(len(batch["base_entity_position_ids"])):
                                gt_weight[i, -1, batch["base_entity_position_ids"][i]] = 0.0
                                gt_weight[i, batch["source_entity_position_ids"][i], batch["base_entity_position_ids"][i]] = 1.0
                                                        
                            self.weight_loss = 50 * torch.nn.functional.mse_loss(intervention_weight, gt_weight)
                        else:
                            self.weight_loss = 0.0
                    else:
                        self.weight_loss = 0.0
                        
                    self.training_loss = self.prediction_loss + self.weight_loss
                    
                    self.training_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(), 4.0
                    )  # just implemented this! dunno if a cap of 1 to large, so I'm messing with reducing it

                    # Check for nan gradients
                    # if check_nan_gradients(self):
                    #     break

                    # Backwards step
                    self.opt.step()

                    # metrics
                    epoch_train_loss += self.training_loss.item() * current_batch_size
                    gradients = [
                        p.grad.view(-1) for p in self.parameters() if p.grad is not None
                    ]
                    all_gradients = torch.cat(gradients)
                    gradient_norm = torch.norm(all_gradients).item()
                    epoch_gradient_norm += gradient_norm * current_batch_size

                    metrics = {
                        "step": cur_steps,
                        "train_batch_total_loss": self.training_loss.item(),
                        "train_batch_prediction_loss": self.prediction_loss.item(),
                        "train_batch_gradient_norm": gradient_norm,
                    }

                    if wandb.run:
                        wandb.log(metrics)
                    if cur_steps % 5 == 0:
                        print(metrics)

                    # Update progress bar
                    pbar.update(1)  # note: this was incorrectly displaying before!
                    cur_steps += 1
                    
                if wandb.run:
                    wandb.log(
                        {
                            "epoch_train_total_loss": epoch_train_loss
                            / num_datapoints_in_epoch,
                            "gradient_norm": epoch_gradient_norm
                            / num_datapoints_in_epoch,
                        }
                    )
        # Save the final model
        if save_dir is not None:
            self.save_model(os.path.join(save_dir, "final_model.pt"))
