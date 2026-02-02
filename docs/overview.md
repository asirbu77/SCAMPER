The following document provides a brief overview of the `SCAMPER` package.

## Higher level design
SCAMPER has been designed to be as flexible as possible, allowing the user to train models with different network architectures, learning algorithms, and a variety of different experiment configurations. It is built to support *in virtual neuroscience experiments* across different bio-mechanically realistic animal models and motion capture registrations that the user specifies. The following sections will provide a brief overview of the higher level structure of the package:


```mermaid
graph TD
    scamper -- training scripts and logging --> agent
    scamper -- env / task / walker creation  --> environment
    scamper -- input preprocessing / model checkpoint --> io

    subgraph " "
    agent --> logging
    agent --> RL-training-scripts
    end

    subgraph " "
    environment --> wrappers
    environment --> walker
    environment --> tasks
    end

    subgraph " "
    io --> checkpoint
    io --> preprocess
    end

```

## Training pipeline & software stacks

```
```
