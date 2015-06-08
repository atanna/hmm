class Track():
    def __init__(self, chr_i, sample_pos, arr_data, path, bin_size=200):
        self.chr_i = chr_i
        self.sample_pos = sample_pos
        self.arr_data = arr_data
        self.path = path
        self.bin_size = bin_size
        self.first_line = "browser position chr{}:{}-{}\n" \
            .format(self.chr_i,
                    self.sample_pos[0][0],
                    self.sample_pos[0][-1])
        self.max_score = 1000

    def _push_peak(self, state, start_pos, end_pos, f, max_score=1000,
                   peak_name="", color0=(0, 255, 0), color1=(255, 0, 0)):
        if state:
            color = color1
        else:
            color = color0

        f.write(
            "chr{chr_i} {chromStart} {chromEnd} {name} {score} + {chromStart} "
            "{chromEnd} {rgb[0]},{rgb[1]},{rgb[2]}\n"
                .format(chr_i=self.chr_i,
                        chromStart=start_pos,
                        chromEnd=end_pos,
                        name=peak_name,
                        score=max_score,
                        rgb=color))

    def create_track_peaks(self, name, states, priority=0):
        with open("{}{}_peaks.bed".format(self.path, name), "wt") as f:
            f.write(self.first_line)
            f.write("browser hide all\n")
            f.write("track name=\"{name}_peaks\" "
                    "description=\"{name}\" "
                    "itemRgb=\"On\" "
                    "priority={priority}\n"
                    .format(name=name, priority=priority))
            for i, st in enumerate(states):
                start_pos = self.sample_pos[i][0]
                end_pos = start_pos + self.bin_size - 1
                curr_st = st[0]
                for j, pos in enumerate(self.sample_pos[i]):
                    if st[j] == curr_st:
                        end_pos += self.bin_size
                        continue
                    self._push_peak(curr_st, start_pos, end_pos, f,
                                    peak_name="{}_{}".format(i, j))
                    start_pos = pos
                    end_pos = start_pos + self.bin_size - 1
                    curr_st = st[j]
                self._push_peak(curr_st, start_pos, end_pos, f,
                                peak_name="{}_{}".format(i, j))

    def create_track_data(self, xmax, priority=19):
        print("xmax", xmax)
        with open("{}data.bed".format(self.path), "wt") as f:
            f.write(self.first_line)
            f.write("track name=\"ChIP peaks\" description=\""
                    "normalized data\" useScore=1 priority={}\n".format(
                priority))
            for i, data in enumerate(self.arr_data):
                for j, pos in enumerate(self.sample_pos[i]):
                    f.write(
                        "chr{chr_i} {chromStart} {chromEnd} {name} {score}\n"
                            .format(chr_i=self.chr_i,
                                    chromStart=pos,
                                    chromEnd=pos + self.bin_size,
                                    name="{}_{}".format(i, j),
                                    score=int(
                                        data[j] * self.max_score / xmax)))

    def create_track_data_bar(self, name="Data", e_state_1=0, priority=20):
        with open("{}data_bar.txt".format(self.path), "wt") as f:
            f.write(self.first_line)
            f.write("track type=wiggle_0 name=\"{}\" "
                    "description=\"data\" visibility=full "
                    "autoScale=on "
                    "yLineMark={} yLineOnOff=on "
                    "color=15,15,15 "
                    "priority={}\n".format(name, e_state_1, priority))
            for i, data in enumerate(self.arr_data):
                f.write("fixedStep chrom=chr{} start={} step={} span={}\n"
                        .format(self.chr_i, self.sample_pos[i][0],
                                self.bin_size, self.bin_size - 1))
                for x in data:
                    f.write("{}\n".format(int(x)))
