if True:
    fps = 30
    rPPGNP = rPPG.detach().numpy()
    rPPGNP = np.transpose(rPPGNP)
    BVP_labelNP = BVP_label.detach().numpy()
    pulse_BVP_labelNP = get_pulse.get_rfft_pulse(BVP_labelNP, fps)  # get pulse from signal
    pulse_PPGNP = get_pulse.get_rfft_pulse(rPPGNP, fps)  # get pulse from signal
    max_time = rPPGNP.size / fps
    time_steps = np.linspace(0, max_time, rPPGNP.size)
    plt.figure(figsize=(15, 15))
    plt.title('EPOCH {}:'.format(n + 1))
    plt.plot(time_steps, rPPGNP, label='rPPG')
    plt.plot(time_steps, BVP_labelNP, label='BVP_label')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    print('Label Puls {} Test result {}'.format(pulse_BVP_labelNP, pulse_PPGNP))
    n = n + 1