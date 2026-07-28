// Noms d'affichage des bots — source unique de vérité côté client.
//
// La base et le serveur ne manipulent que des *clés* d'agent (« dede »,
// « doudou », « oracle_dd ») : ce sont elles qui nomment un spec, une ligne
// `games.agents` ou une entité Elo, et elles ne doivent pas bouger. Le nom
// lisible, lui, n'existe qu'à l'écran, et une même clé doit s'y rendre
// pareil dans le Salon et dans Rejouer.
//
// Version courte volontairement : `agents.AGENT_NAMES` côté serveur dit
// « Dédé (IS-DD) », ce qui décrit l'algorithme. À une table on lit un siège,
// pas une architecture.
export const BOT_LABELS = { dede: 'Dédé', doudou: 'DouDou', oracle_dd: 'Oracle' };

/** Nom lisible d'un bot. Une clé inconnue se rend telle quelle. */
export function botLabel(key) {
    return BOT_LABELS[key] || key || 'Bot';
}
