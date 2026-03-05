package ai.mcts.submissions.nick_mcts;

import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.LanchesterEvaluationFunction;
import ai.mcts.naivemcts.NaiveMCTS;
import java.util.ArrayList;
import java.util.List;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;
import rts.units.Unit;
import rts.PhysicalGameState;

/**
 * NickMCTS: An adaptive NaiveMCTS agent that uses a local LLM (Ollama)
 * to dynamically adjust evaluation weights based on game state.
 */
public class NickMCTS extends NaiveMCTS {
    private UnitTypeTable utt;
    private final StrategyController controller = new StrategyController();
    private int lastUpdateFrame = -1;

    public NickMCTS(UnitTypeTable utt) {
        // Initializing with your custom MyEvaluation
        super(100, -1, 100, 10, 0.3f, 0.0f, 0.4f,
              new WorkerRush(utt), 
              new MyEvaluation(utt, null), // Controller assigned below
              true);
        this.utt = utt;
        ((MyEvaluation)this.ef).setController(this.controller);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        // Consult the "General" (LLM) every 400 frames
        if (gs.getTime() % 400 == 0 && gs.getTime() != lastUpdateFrame) {
            lastUpdateFrame = gs.getTime();
            controller.updateStrategy(gs, player);
        }
        return super.getAction(player, gs);
    }

    @Override
    public AI clone() {
        return new NickMCTS(utt);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}

/**
 * Manages asynchronous communication with Ollama.
 * Uses a non-blocking approach to ensure the tournament clock never times out.
 */
class StrategyController {
    // Volatile ensures the MCTS threads see the latest updates from the LLM thread
    public volatile float aggression = 1.0f;
    public volatile float threatWeight = 1.0f;
    public volatile float resourceWeight = 0.2f;

    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(500))
            .build();

    public void updateStrategy(GameState gs, int player) {
        String stateSummary = summarizeState(gs, player);
        String prompt = "MicroRTS state: " + stateSummary + 
                        ". Respond ONLY JSON: {\"agg\":float(0.5-2), \"thr\":float(0-5), \"res\":float(0-1)}";

        String jsonBody = "{\"model\": \"llama3\", \"prompt\": \"" + prompt + "\", \"stream\": false, \"format\": \"json\"}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:11434/api/generate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
              .thenApply(HttpResponse::body)
              .thenAccept(this::parseAndApply)
              .exceptionally(e -> null); // Silently fail if Ollama is missing
    }

    private String summarizeState(GameState gs, int player) {
        int[] my = countUnits(gs, player);
        int[] en = countUnits(gs, 1 - player);
        return String.format("Me[W:%d,C:%d,B:%d,Br:%d,G:%d] En[W:%d,C:%d,B:%d,Br:%d,G:%d]",
            my[0], my[1], my[2], my[3], gs.getPlayer(player).getResources(),
            en[0], en[1], en[2], en[3], gs.getPlayer(1-player).getResources());
    }

    private int[] countUnits(GameState gs, int p) {
        int w=0, c=0, b=0, br=0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == p) {
                String n = u.getType().name;
                if (n.equals("Worker")) w++;
                else if (n.equals("Base")) b++;
                else if (n.equals("Barracks")) br++;
                else c++;
            }
        }
        return new int[]{w, c, b, br};
    }

    private void parseAndApply(String response) {
        try {
            // Manual extraction to avoid external JSON libraries
            this.aggression = extract(response, "\"agg\"");
            this.threatWeight = extract(response, "\"thr\"");
            this.resourceWeight = extract(response, "\"res\"");
        } catch (Exception e) { /* Keep current weights */ }
    }

    private float extract(String json, String key) {
        int i = json.indexOf(key);
        int start = json.indexOf(":", i) + 1;
        int end = json.indexOf(",", start);
        if (end == -1) end = json.indexOf("}", start);
        return Float.parseFloat(json.substring(start, end).replace("\"", "").trim());
    }
}

class MyEvaluation extends LanchesterEvaluationFunction {
    private StrategyController sc;

    public MyEvaluation(UnitTypeTable utt, StrategyController sc) {
        this.sc = sc;
    }

    public void setController(StrategyController sc) { this.sc = sc; }

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        // If controller is null or Ollama fails, use reasonable defaults
        float agg = (sc != null) ? sc.aggression : 1.0f;
        float thr = (sc != null) ? sc.threatWeight : 1.0f;
        float res = (sc != null) ? sc.resourceWeight : 0.2f;

        float score = super.evaluate(maxplayer, minplayer, gs) * agg;
        float threatPenalty = calculateThreat(maxplayer, gs) * thr;
        
        float carryingBonus = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == maxplayer && u.getResources() > 0) {
                carryingBonus += res;
            }
        }
        return score - threatPenalty + carryingBonus;
    }

    private float calculateThreat(int player, GameState gs) {
        float threatPenalty = 0.0f;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().name.equals("Base")) {
                for (Unit e : pgs.getUnits()) {
                    if (e.getPlayer() == 1 - player) {
                        int dist = Math.abs(u.getX() - e.getX()) + Math.abs(u.getY() - e.getY());
                        if (dist < 8) threatPenalty += (8 - dist) * 0.1f;
                    }
                }
            }
        }
        return threatPenalty;
    }
}
