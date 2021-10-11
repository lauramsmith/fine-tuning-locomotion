SAC_CONFIGS = {
    "Ant-v2":
    {
        "actor_net": "fc_2layers_256units",
        "critic_net": "fc_2layers_256units",

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.2,
        
        "critic_stepsize": 0.0003,
        "critic_batch_size": 256,
        "critic_steps": 256,

        "discount": 0.99,
        "samples_per_iter": 512,
        "replay_buffer_size": 1000000,
        "normalizer_samples": 300000,
        
        "num_action_samples": 1,
        "tar_stepsize": 0.01,
        "steps_per_tar_update": 1,
        "init_samples": 25000
    },
    "Hopper-v2":
    {
        "actor_net": "fc_2layers_256units",
        "critic_net": "fc_2layers_256units",

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.2,
        
        "critic_stepsize": 0.0003,
        "critic_batch_size": 256,
        "critic_steps": 256,

        "discount": 0.99,
        "samples_per_iter": 512,
        "replay_buffer_size": 1000000,
        "normalizer_samples": 300000,
        
        "num_action_samples": 1,
        "tar_stepsize": 0.01,
        "steps_per_tar_update": 1,
        "init_samples": 25000
    },
    "HalfCheetah-v2":
    {
        "actor_net": "fc_2layers_256units",
        "critic_net": "fc_2layers_256units",

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.2,
        
        "critic_stepsize": 0.0003,
        "critic_batch_size": 256,
        "critic_steps": 256,

        "discount": 0.99,
        "samples_per_iter": 512,
        "replay_buffer_size": 1000000,
        "normalizer_samples": 300000,
        
        "num_action_samples": 1,
        "tar_stepsize": 0.01,
        "steps_per_tar_update": 1,
        "init_samples": 25000
    },
    "Walker2d-v2":
    {
        "actor_net": "fc_2layers_256units",
        "critic_net": "fc_2layers_256units",

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.2,
        
        "critic_stepsize": 0.0003,
        "critic_batch_size": 256,
        "critic_steps": 256,

        "discount": 0.99,
        "samples_per_iter": 512,
        "replay_buffer_size": 1000000,
        "normalizer_samples": 300000,
        
        "num_action_samples": 1,
        "tar_stepsize": 0.01,
        "steps_per_tar_update": 1,
        "init_samples": 25000
    },
    "Humanoid-v2":
    {
        "actor_net": "fc_2layers_256units",
        "critic_net": "fc_2layers_256units",

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.2,
        
        "critic_stepsize": 0.0003,
        "critic_batch_size": 256,
        "critic_steps": 256,

        "discount": 0.99,
        "samples_per_iter": 512,
        "replay_buffer_size": 1000000,
        "normalizer_samples": 300000,
        
        "num_action_samples": 1,
        "tar_stepsize": 0.01,
        "steps_per_tar_update": 1,
        "init_samples": 25000
    },
    "A1-Motion-Imitation-REDQ-Pretrain":
    {
        "actor_net": "fc_2layers_512units",
        "critic_net": "fc_2layers_512units",
        "use_MPI_solver": False, 
        "parallel_ensemble": True,

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 512,
        "actor_steps": 512,
        "action_std": 0.15,

        "num_critic_nets": 10,
        "critic_stepsize": 0.0003,
        "critic_batch_size": 512,
        "critic_steps": 20*512,
        "num_ensemble_subset": 2,

        "discount": 0.95,
        "samples_per_iter": 512,
        "replay_buffer_size": int(1e6),
        "normalizer_samples": 30000,
        "enable_val_norm": False,

        "num_action_samples": 1,
        "tar_stepsize": 5e-3,
        "steps_per_tar_update": 1,
        "init_samples": 20000
    },
    "A1-Motion-Imitation-REDQ-Finetune":
    {
        "actor_net": "fc_2layers_512units",
        "critic_net": "fc_2layers_512units",
        "use_MPI_solver": False, 
        "parallel_ensemble": True,

        "actor_stepsize": 0.0001,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.15,

        "num_critic_nets": 10,
        "critic_stepsize": 0.0001,
        "critic_batch_size": 256,
        "critic_steps": 2*256,
        "num_ensemble_subset": 2,

        "discount": 0.95,
        "samples_per_iter": 512,
        "replay_buffer_size": int(1e6),
        "normalizer_samples": 0,
        "enable_val_norm": False,

        "num_action_samples": 1,
        "tar_stepsize": 1e-3,
        "steps_per_tar_update": 1,
        "init_samples": 5000
    },
    "A1-Motion-Imitation-Vanilla-SAC-Pretrain":
    {
        "actor_net": "fc_2layers_512units",
        "critic_net": "fc_2layers_512units",
        "use_MPI_solver": False, 
        "parallel_ensemble": True,

        "actor_stepsize": 0.0003,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 512,
        "actor_steps": 512,
        "action_std": 0.15,

        "num_critic_nets": 2,
        "critic_stepsize": 0.0003,
        "critic_batch_size": 512,
        "critic_steps": 512,
        "num_ensemble_subset": 2,

        "discount": 0.95,
        "samples_per_iter": 512,
        "replay_buffer_size": int(1e6),
        "normalizer_samples": 30000,
        "enable_val_norm": False,

        "num_action_samples": 1,
        "tar_stepsize": 5e-3,
        "steps_per_tar_update": 1,
        "init_samples": 20000
    },
    "A1-Motion-Imitation-Vanilla-SAC-Finetune":
    {
        "actor_net": "fc_2layers_512units",
        "critic_net": "fc_2layers_512units",
        "use_MPI_solver": False, 
        "parallel_ensemble": True,

        "actor_stepsize": 0.0001,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 256,
        "action_std": 0.15,

        "num_critic_nets": 2,
        "critic_stepsize": 0.0001,
        "critic_batch_size": 256,
        "critic_steps": 256,
        "num_ensemble_subset": 2,

        "discount": 0.95,
        "samples_per_iter": 512,
        "replay_buffer_size": int(1e6),
        "normalizer_samples": 0,
        "enable_val_norm": False,

        "num_action_samples": 1,
        "tar_stepsize": 1e-3,
        "steps_per_tar_update": 1,
        "init_samples": 5000
    }
}