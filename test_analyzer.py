"""Sanity checks on a few hand-written samples. This is NOT a benchmark:
the samples were written for calibration, so real accuracy needs your own
labelled data. Run: python test_analyzer.py"""
from analyzer import analyze_text

SAMPLES = {
"ai_essay": ("""In today's fast-paced world, effective time management plays a crucial role in achieving both personal and professional success. It is important to note that time management is not just about completing tasks, but about prioritizing what truly matters. By adopting the right strategies, individuals can navigate the complexities of modern life with confidence.

Moreover, there are various techniques that can enhance productivity. For example, the Pomodoro Technique encourages focused work, regular breaks, and consistent progress. Additionally, setting clear goals fosters accountability, motivation, and a sense of direction. Furthermore, leveraging digital tools can streamline workflows and ensure seamless collaboration.

However, it is worth noting that no single approach works for everyone. Different people may have different preferences, and it is often beneficial to experiment with a range of methods. By understanding their own habits, individuals can develop a holistic strategy that supports long-term growth.

In conclusion, mastering time management is a multifaceted journey that requires patience, discipline, and reflection. Ultimately, those who invest in these skills will be well-equipped to thrive in an ever-evolving landscape.""", (70, 100)),
"ai_marketing_short": ("""Unlock the power of seamless collaboration with our cutting-edge platform. Whether you're a startup, a growing team, or a global enterprise, our robust solution streamlines workflows, enhances productivity, and fosters innovation. Embark on a journey toward success today. It's not just software, it's a testament to what teams can achieve together.""", (55, 100)),
"ai_residue": ("""Certainly! Here is a revised version of your paragraph: The report shows steady growth across all regions. Revenue increased in the second quarter. Costs remained stable over the same period. I hope this helps!""", (60, 100)),
"ai_plain": ("""Remote work has changed how companies operate. Employees can work from anywhere, which gives them more flexibility. Companies can hire from a wider pool of talent. However, remote work also brings challenges. Communication can be harder without face-to-face interaction. Managers need to find new ways to build trust. Overall, remote work offers many benefits, but it requires careful planning.""", (40, 100)),
"human_blog": ("""I missed my train on Tuesday because I stopped to argue with a pigeon. Not proud of it. The thing had my sandwich, half of it anyway, and I wasn't going to let that go.

Anyway, I got to the office forty minutes late and my manager, Priya, just looked at me over her laptop. She didn't say anything. That was worse, honestly. I've had bosses who yell, and you know where you stand, but the quiet look? I spent the whole morning rewriting an email nobody had asked about.

By lunch I'd decided to quit time management for good. Seven years of apps, three notebooks, one very expensive planner from a shop in Leeds. What actually worked was setting my alarm twenty minutes earlier and putting my shoes by the door. That's it. Boring, but I haven't missed a train since, and the pigeon hasn't come back either.""", (0, 25)),
"human_formal": ("""The Phillips curve describes an inverse relationship between the rate of unemployment and the rate of wage inflation in an economy. Originally observed by A. W. Phillips in 1958 using United Kingdom data from 1861 to 1957, the relationship was later generalised to price inflation. Samuelson and Solow presented it as a menu of policy choices in 1960. During the 1970s, the combination of high inflation and high unemployment weakened this interpretation.

Friedman and Phelps argued that the trade-off exists only in the short run. Workers adjust their expectations of inflation, and the curve then shifts upward. In the long run, unemployment returns to its natural rate regardless of the inflation rate. The expectations-augmented Phillips curve therefore became the standard framework in macroeconomics textbooks. Empirical work since the 1990s finds a flatter curve, although estimates differ across countries and periods.

Several explanations have been proposed for this flattening. Central banks with credible inflation targets anchor expectations. Globalisation may also have reduced the sensitivity of domestic prices to domestic slack. Measurement of the natural rate remains uncertain, and revisions to estimates are common.""", (0, 45)),
"human_non_native": ("""My name is Rahul and I am a student of economics. I live in a small town with my family. Every day I wake up at six o'clock in the morning. After that I take a bath and I eat breakfast. Then I go to the college by bus. The college is very big and it has a nice library. I like to read books in the library. In the evening I come back to home and I do my homework. My favourite subject is economics because it is very interesting. I want to become a professor in future. I work hard for this dream every day.""", (0, 45)),
}

if __name__ == "__main__":
    bad = 0
    for name, (text, (lo, hi)) in SAMPLES.items():
        r = analyze_text(text)
        ok = lo <= r["ai_percent"] <= hi
        bad += not ok
        top = ", ".join(s["label"] for s in r["signals"][:3]) or "none"
        print(f"{'ok  ' if ok else 'FAIL'} {name:20s} {r['ai_percent']:3d}%  conf={r['confidence']:6s} expected {lo}-{hi}  | {top}")
    raise SystemExit(1 if bad else 0)