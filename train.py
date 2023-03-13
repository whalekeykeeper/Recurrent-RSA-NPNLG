import torch
import torch.nn as nn
import numpy as np
import os
from train.data_loader import get_loader, Vocabulary
from train.encoder_decoder_models import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COCO_CAPTIONS_JSON = "./data/captions_train2014.json"
VG_CAPTIONS_JSON = "./data/visual_genome_JSON/region_descriptions.json"
TS1_JSON = "./data/test_sets/ts1/ts1.json"
TS2_JSON = "./data/test_sets/ts2/ts2.json"
IMAGE_DIR = "./data/train2014"
MODEL_OUTPUT = "./output"   # Save models

# Hyperparameters
CROP_SIZE = 224     # randomly crops images down to specified size
EMBED_SIZE = 256    # embedding size
HIDDEN_SIZE = 512   # LSTM hidden layer
NUM_LAYERS = 1      # number of layers for LSTM

# Training parameters
# (my hardware is not that great, hence the low numbers)
NUM_EPOCHS = 2
BATCH_SIZE = 12
NUM_WORKERS = 1
LEARNING_RATE = 0.001
VOCAB_SIZE = 20     # dataset size (set to None, to use entire dataset)


if __name__ == '__main__':

    # Create model directory
    if not os.path.exists(MODEL_OUTPUT):
        os.makedirs(MODEL_OUTPUT)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # # TRAIN WITH COCO
    # # Load vocabulary wrapper
    # vocab = Vocabulary(COCO_CAPTIONS_JSON, level='char', dataset='coco', data_size=VOCAB_SIZE) # level='word
    # # Build data loader
    # data_loader = get_loader(IMAGE_DIR, COCO_CAPTIONS_JSON, vocab,
    #                          transform, batch_size=BATCH_SIZE,
    #                          shuffle=True, num_workers=NUM_WORKERS, data_size=VOCAB_SIZE, dataset='coco'
    #                          ts1_json=TS1_JSON, ts2_json=TS2_JSON
    #                          )
    # print(f"LENGTH {len(data_loader)}")

    # TRAIN WITH VG
    # Load vocabulary wrapper
    vocab = Vocabulary(VG_CAPTIONS_JSON, level='char',
                       dataset='vg', data_size=VOCAB_SIZE)  # level='word
    # Build data loader
    data_loader = get_loader(IMAGE_DIR, VG_CAPTIONS_JSON, vocab,
                             transform, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS, data_size=VOCAB_SIZE, dataset='vg', ts1_json=TS1_JSON, ts2_json=TS2_JSON)

    print(f"LENGTH {len(data_loader)}")

    # Build the models
    encoder = EncoderCNN(EMBED_SIZE).to(device)
    decoder = DecoderRNN(embed_size=EMBED_SIZE,
                         hidden_size=HIDDEN_SIZE,
                         vocab_size=len(vocab),
                         num_layers=NUM_LAYERS).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + \
        list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(NUM_EPOCHS):
        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, 2, i, total_step, loss.item(), np.exp(loss.item())))

    # Save the model
    torch.save(decoder.state_dict(), os.path.join("./output", 'decoder.ckpt'))
    torch.save(encoder.state_dict(), os.path.join("./output", 'encoder.ckpt'))
